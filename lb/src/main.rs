use hyper::client::HttpConnector;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Client, Request, Response, Server, Uri};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout, Duration};

/// If the H100 KV cache usage ratio is equal or above this, we route to L40.
const CAPACITY_THRESHOLD: f64 = 0.7;

struct MetricsState {
    h100_kv_ratio: f64,
    h100_online: bool,
}

impl MetricsState {
    fn new() -> Self {
        MetricsState { 
            h100_kv_ratio: 0.0,
            h100_online: true,
        }
    }
}

/// Polls the metrics endpoint on the H100 server every 10 seconds and updates the shared state.
async fn poll_metrics(metrics_state: Arc<RwLock<MetricsState>>) {
    let client = Client::new();
    loop {
        // Metrics endpoint on H100; adjust the URL if needed.
        let req = Request::builder()
            .method("GET")
            .uri("http://0.0.0.0:8002/metrics")
            .body(Body::empty())
            .expect("Failed to build metrics request");
        match client.request(req).await {
            Ok(resp) => {
                match hyper::body::to_bytes(resp.into_body()).await {
                    Ok(body_bytes) => {
                        let metrics_text = String::from_utf8_lossy(&body_bytes);
                        let mut used: Option<u64> = None;
                        let mut max: Option<u64> = None;
                        // Scan through the metrics lines.
                        for line in metrics_text.lines() {
                            // Look for the line with used KV blocks for tensorrt_llm.
                            if line.contains("kv_cache_block_type=\"used\"")
                                && line.contains("model=\"tensorrt_llm\"")
                                && line.contains("version=\"1\"")
                            {
                                if let Some(token) = line.split_whitespace().last() {
                                    if let Ok(val) = token.parse::<u64>() {
                                        used = Some(val);
                                    }
                                }
                            }
                            // Look for the line with max KV blocks for tensorrt_llm.
                            if line.contains("kv_cache_block_type=\"max\"")
                                && line.contains("model=\"tensorrt_llm\"")
                                && line.contains("version=\"1\"")
                            {
                                if let Some(token) = line.split_whitespace().last() {
                                    if let Ok(val) = token.parse::<u64>() {
                                        max = Some(val);
                                    }
                                }
                            }
                        }
                        if let (Some(used_val), Some(max_val)) = (used, max) {
                            let ratio = used_val as f64 / max_val as f64;
                            println!(
                                "Polled KV Cache: used: {}, max: {}, ratio: {:.2}",
                                used_val, max_val, ratio
                            );
                            let mut state = metrics_state.write().await;
                            state.h100_kv_ratio = ratio;
                            state.h100_online = true; // Metrics successful, mark H100 as online.
                        } else {
                            println!("Could not parse KV cache metrics");
                            let mut state = metrics_state.write().await;
                            state.h100_online = false;
                        }
                    }
                    Err(e) => {
                        println!("Failed to read metrics body: {}", e);
                        let mut state = metrics_state.write().await;
                        state.h100_online = false;
                    }
                }
            }
            Err(e) => {
                println!("Metrics request error: {}", e);
                let mut state = metrics_state.write().await;
                state.h100_online = false;
            }
        }
        sleep(Duration::from_secs(10)).await;
    }
}

/// Forwards the request to the appropriate backend based on the current metrics state.
async fn route_request(
    mut req: Request<Body>,
    metrics_state: Arc<RwLock<MetricsState>>,
) -> Result<Response<Body>, hyper::Error> {

    let whole_body = hyper::body::to_bytes(req.body_mut()).await?;
    

    let use_l40 = {
        let state = metrics_state.read().await;
        if !state.h100_online {
            println!("H100 is offline. Routing to L40.");
            true
        } else if state.h100_kv_ratio >= CAPACITY_THRESHOLD {
            println!("Routing to L40 due to high H100 KV usage.");
            true
        } else {
            println!("Routing to H100.");
            false
        }
    };

    let backend_base = if use_l40 {
        "http://192.168.1.13:8003/v2/models/tensorrt_llm_bls/generate"
    } else {
        "http://192.168.1.18:8000/v2/models/ensemble/generate"
    };


    //let target_path = "/v2/models/ensemble/generate";
    let new_uri_str = format!("{}", backend_base); //, {} target_path);
    let new_uri = Uri::from_str(&new_uri_str).expect("Failed to parse new URI");

    let mut builder = Request::builder().method(req.method()).uri(new_uri);
    for (key, value) in req.headers().iter() {
        builder = builder.header(key, value);
    }
    let new_req = builder
        .body(Body::from(whole_body))
        .expect("Failed to build new request");

    let client: Client<HttpConnector, Body> = Client::new();
    let resp = match timeout(Duration::from_secs(500), client.request(new_req)).await {
        Ok(result) => result,
        Err(_) => Ok(Response::builder()
            .status(504)
            .body(Body::from("Backend timeout"))
            .unwrap()),
    }?;

    Ok(resp)
}

#[tokio::main]
async fn main() {

    let metrics_state = Arc::new(RwLock::new(MetricsState::new()));


    let metrics_state_clone = metrics_state.clone();
    tokio::spawn(async move {
        poll_metrics(metrics_state_clone).await;
    });


    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));


    let make_svc = make_service_fn(move |_conn| {
        let metrics_state = metrics_state.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |req| {
                route_request(req, metrics_state.clone())
            }))
        }
    });

    let server = Server::bind(&addr).serve(make_svc);
    println!("Rust load balancer listening on http://{}", addr);

    if let Err(e) = server.await {
        eprintln!("Server error: {}", e);
    }
}
