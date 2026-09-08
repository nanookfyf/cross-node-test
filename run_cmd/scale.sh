
curl -X POST http://127.0.0.1:8005/flash_epscale   -H 'Content-Type: application/json'   -d '{"ep_size": 1, "level": 2}'
curl -X POST http://127.0.0.1:8005/flash_epscale   -H 'Content-Type: application/json'   -d '{"ep_size": 2, "level": 2}'


curl -X POST "http://127.0.0.1:8005/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer EMPTY"   -d '{
    "model": "/mnt/nvme/fyf/models/DeepSeek-V2-Lite",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "max_tokens": 10
  }'