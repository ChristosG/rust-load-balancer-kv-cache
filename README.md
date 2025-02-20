Setup: 2 servers:
    - one with 2x H100
    - one with 2x L40s


Having the one with the H100 as the main server, with a triton-inference-server and an tensorrt-llm model deployed,
i noticed that when its kv-cache (due to multiple concurrent requests) reaches its limit, the response has much more delay.
This is an attempt to de-digest the kv-cache by routing to the less powerful server until kv-kache of the main server free up until a threshold.

Can also be used as just an entrypoint for routing requests to different server instances.
