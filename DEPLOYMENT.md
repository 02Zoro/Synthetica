# Synthetica Deployment Guide

This guide provides simple deployment options for your Synthetica AI research assistant.

## 🚀 Quick Deployment Options

### Option 1: Demo Mode (Simplest - 2 minutes)
Perfect for demonstrations and testing:

```bash
# Start demo mode
docker-compose --profile demo-mode up -d

# Access your application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8001
```

### Option 2: ML-Enhanced Mode (5 minutes)
Full AI/ML capabilities with knowledge graphs:

```bash
# Start ML mode
docker-compose --profile ml-mode up -d

# Access your application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8001
# Neo4j: http://localhost:7474
```

### Option 3: Full Production Mode (10 minutes)
Complete system with all databases:

```bash
# Start full mode
docker-compose --profile full-mode up -d

# Access your application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# Neo4j: http://localhost:7474
# PostgreSQL: localhost:5432
```

## 🌐 Public Deployment

### Railway (Recommended)
1. Push your code to GitHub
2. Go to [railway.app](https://railway.app)
3. Connect your GitHub repository
4. Railway will auto-detect your `docker-compose.yml`
5. Deploy with one click
6. Get your public URL instantly

### Render (Alternative)
1. Go to [render.com](https://render.com)
2. Create "Web Service"
3. Connect your GitHub repository
4. Set build command: `docker-compose --profile demo-mode up -d`
5. Deploy

### Vercel (Frontend Only)
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Set build command: `cd frontend && npm run build`
4. Deploy

## 📋 Deployment Commands

### Local Development
```bash
# Demo mode (fastest)
docker-compose --profile demo-mode up -d

# ML mode (with Neo4j)
docker-compose --profile ml-mode up -d

# Full mode (all databases)
docker-compose --profile full-mode up -d
```

### Production Deployment
```bash
# Build and start
docker-compose --profile demo-mode up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Clean up
docker-compose down -v
```

## 🔧 Environment Configuration

### Demo Mode
No additional configuration needed - works out of the box!

### ML Mode
Requires Neo4j database (automatically started with docker-compose).

### Full Mode
Requires all databases (PostgreSQL, Neo4j, Redis).

## 📊 Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 | React application |
| Backend (Demo) | 8001 | FastAPI demo server |
| Backend (ML) | 8001 | FastAPI ML server |
| Backend (Full) | 8000 | FastAPI full server |
| Neo4j | 7474 | Graph database UI |
| Neo4j Bolt | 7687 | Graph database connection |
| PostgreSQL | 5432 | Relational database |
| Redis | 6379 | Cache database |

## 🚨 Troubleshooting

### Port Already in Use
```bash
# Find process using port
netstat -tulpn | grep :3000

# Kill process
kill -9 <PID>
```

### Services Not Starting
```bash
# Check logs
docker-compose logs synthetica-backend-demo

# Restart services
docker-compose restart
```

### Database Connection Issues
```bash
# Check if databases are running
docker-compose ps

# Restart specific service
docker-compose restart neo4j
```

## 🎯 Quick Start for Demo

1. **Clone and navigate to project**
   ```bash
   cd project-Hypothisis
   ```

2. **Start demo mode**
   ```bash
   docker-compose --profile demo-mode up -d
   ```

3. **Open your browser**
   - Go to http://localhost:3000
   - Test the research functionality

4. **Stop when done**
   ```bash
   docker-compose down
   ```

## 🌍 Public URL Options

### For Immediate Public Access
Use **ngrok** for instant public URL:

```bash
# Install ngrok
npm install -g ngrok

# Start your app
docker-compose --profile demo-mode up -d

# Create public tunnel
ngrok http 3000

# Use the provided URL (e.g., https://abc123.ngrok.io)
```

### For Permanent Deployment
Use **Railway** or **Render** for permanent public hosting.

## 📝 Notes

- **Demo Mode**: Fastest setup, no databases required
- **ML Mode**: Includes Neo4j for knowledge graphs
- **Full Mode**: Complete system with all databases
- All modes include the React frontend
- Services are configured with proper health checks
- Volumes are mounted for data persistence

Choose the mode that best fits your needs!
