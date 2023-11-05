echo "Running: curl http://localhost:8080/hello"
curl http://localhost:8080/hello
echo ""
echo "Running: docker logs my-flask-container"
docker logs my-flask-container
echo ""
echo "Running: curl -X POST http://localhost:8080/chat -H with options"
curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message": "Hello, how are you?"}'
echo ""
echo "Running: docker images"
docker images
echo ""
echo "Running: docker ps -a"
docker ps -a