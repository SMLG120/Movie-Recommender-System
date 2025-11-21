# Server Deployment Guide

## Movie Recommender API - McGill CS Server Deployment

Deploy your movie recommendation system to `fall2025-comp585-6.cs.mcgill.ca`

## 🚀 Server Deployment Steps

### 1. Upload Code to Server
```bash
# From local machine, upload your code
scp -r . team-6@fall2025-comp585-6.cs.mcgill.ca:~/team-6/
```

### 2. SSH to Server and Deploy
```bash
# SSH to your team server
ssh team-6@fall2025-comp585-6.cs.mcgill.ca

# Navigate to project
cd team-6

# Build Docker image
docker build -f docker/Dockerfile -t movie-recommender:v1.0 .

# Run with logging limits (required)
docker run -it --log-opt max-size=50m --log-opt max-file=5 -p 8080:8080 movie-recommender:v1.0
```

### 3. Test Deployment
```bash
# Test from server
curl http://localhost:8080/recommend/1

# Test external access
curl http://fall2025-comp585-6.cs.mcgill.ca:8080/recommend/1
```

## 📋 Production Requirements Met

### ✅ **API Endpoint Specification**
- **URL**: `http://fall2025-comp585-6.cs.mcgill.ca:8080/recommend/{USER_ID}`
- **Method**: GET
- **Response**: Plain text, comma-separated movie IDs
- **Format**: `"movie_id1,movie_id2,movie_id3,..."`
- **Limit**: Up to 20 recommendations
- **Order**: First ID = highest recommendation

### ✅ **Docker Configuration**
- **Port**: 8080 (production requirement)
- **Logging**: Limited with `--log-opt max-size=50m --log-opt max-file=5`
- **Environment**: Production-ready with proper error handling

### ✅ **Server Integration**
- **External APIs**: Uses `http://fall2025-comp585.cs.mcgill.ca:8080/user/{ID}` and `/movie/{ID}`
- **Kafka**: Ready for Kafka integration (stream from `movielog6`)
- **VPN**: Accessible only through McGill VPN

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HTTP Client   │    │   Flask API      │    │ RecommenderEngine│
│                 │    │   (port 8080)    │    │                 │
│ GET /recommend/ │ -> │                  │ -> │  - Load Model   │
│    <user_id>    │    │  - Route Handler │    │  - Get User Info│
│                 │    │  - Error Handle  │    │  - Generate Recs│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Comma-separated │    │    Logging &     │    │   External APIs │
│   Movie IDs     │    │  Error Recovery  │    │ User/Movie Data │
│ "1,42,123,..."  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Key Features

### **Production-Ready API**
- **Port 8080**: Meets deployment requirement
- **Plain text response**: Comma-separated movie IDs (not JSON)
- **Error handling**: Fallback recommendations if engine fails  
- **Logging**: Comprehensive logging for debugging
- **Health checks**: `/health` endpoint for monitoring

### **Robust Model Loading**
- **Multiple paths**: Tries Docker, local, and alternative paths
- **HuggingFace fallback**: Attempts HF model first, falls back to local
- **Error recovery**: Provides fallback recommendations if model fails
- **Path flexibility**: Works in both development and production

### **Docker Optimization**
- **Correct port**: Exposes 8080 (not 8082)
- **Log limits**: Prevents disk space issues
- **Environment**: Production environment variables
- **Directory structure**: Creates required paths

## 📝 API Usage Examples

### **Primary Endpoint**
```bash
# Get recommendations for user 123
curl http://fall2025-comp585-6.cs.mcgill.ca:8080/recommend/123
# Response: "42,1,567,23,89,445,12,993,76,234,445,667,88,991,23,45,67,89,12,34"
```

### **Health Check**
```bash
# Check if service is running
curl http://fall2025-comp585-6.cs.mcgill.ca:8080/health
# Response: "OK" (if working) or "Service Degraded" (if issues)
```

### **Service Status**
```bash
# Root endpoint
curl http://fall2025-comp585-6.cs.mcgill.ca:8080/
# Response: "Movie Recommender API - Service Running"
```

## 🐛 Troubleshooting

### **Container Won't Start**
```bash
# Check if model/data files exist
docker run -it movie-recommender:v1.0 ls -la /app/src/models/
docker run -it movie-recommender:v1.0 ls -la /app/data/raw_data/

# Check logs
docker logs <container_id>
```

### **API Returns Fallback Recommendations**
- Check if model file exists in container
- Verify movies.csv is accessible
- Check container logs for initialization errors

### **Port Issues**
```bash
# Verify port mapping
docker ps

# Test locally first
curl http://localhost:8080/recommend/1
```

## 🔍 Monitoring

### **Check Service Status**
```bash
# Monitor Kafka logs for your responses
docker run -it --log-opt max-size=50m --log-opt max-file=5 bitnami/kafka kafka-console-consumer.sh --bootstrap-server fall2025-comp585.cs.mcgill.ca:9092 --topic movielog6
```

### **Docker Management**
```bash
# Clean up old containers/images
docker system prune

# List running containers
docker ps

# Stop container
docker stop <container_id>
```

## 🚨 Important Notes

1. **Port 8080**: Must use port 8080 (not 8082) for production
2. **Plain Text**: Response must be plain text, not JSON
3. **20 Recommendations**: Return exactly up to 20 movie IDs
4. **Comma-separated**: No spaces, just commas between IDs
5. **Logging Limits**: Always use `--log-opt` flags to prevent disk issues
6. **VPN Required**: Only accessible through McGill VPN

## 🎯 Deployment Checklist

- [ ] Model file (`xgb_recommender.joblib`) exists in `src/models/`
- [ ] Movies data (`movies.csv`) exists in `data/raw_data/`
- [ ] Docker image builds without errors
- [ ] Container runs on port 8080
- [ ] API responds with comma-separated movie IDs
- [ ] Health check endpoint works
- [ ] Logging limits are set
- [ ] Service handles errors gracefully

Your movie recommendation system is now production-ready! 🎬🚀