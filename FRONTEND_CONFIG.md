# Frontend Configuration Summary

## Hardcoded Values Found

### 1. ResearchPage.js
- **Line 72**: `http://localhost:8000/api/v1/research/generate`
- **Line 95**: `http://localhost:8000/api/v1/research/async`
- **Line 116**: `http://localhost:8000/api/v1/research/status/${id}`
- **Line 123**: `http://localhost:8000/api/v1/research/result/${id}`

### 2. KnowledgeGraphPage.js
- **Line 81**: `http://localhost:8000/api/v1/research/entities/${encodeURIComponent(searchTerm)}`
- **Line 97**: `http://localhost:8000/api/v1/research/paths/${encodeURIComponent(entity1)}/${encodeURIComponent(entity2)}`

### 3. nginx.conf
- **Line 10**: `listen 3000;` (frontend port)
- **Line 23**: `proxy_pass http://synthetica-backend-full:8000;` (backend container)

### 4. Dockerfile
- **Line 29**: `EXPOSE 3000` (frontend port)

## Configuration Files Created

### 1. frontend.env
Environment variables file with all configuration values:
- API base URLs (localhost and container)
- API endpoints
- Frontend configuration
- Environment settings

### 2. frontend/src/config/api.js
Centralized API configuration utility:
- Environment-based URL selection
- Helper functions for URL construction
- Centralized endpoint management

## Next Steps

1. **Update React components** to use the new configuration:
   - Import `API_CONFIG` from `./config/api.js`
   - Replace hardcoded URLs with `API_CONFIG.getFullUrl('ENDPOINT_NAME')`

2. **Environment-specific configuration**:
   - Development: Use `localhost:8000`
   - Production: Use `synthetica-backend-full:8000`

3. **Docker configuration**:
   - Copy `.env` file to container
   - Set environment variables in docker-compose.yml

## Example Usage

```javascript
import API_CONFIG from './config/api';

// Instead of: 'http://localhost:8000/api/v1/research/generate'
const url = API_CONFIG.getFullUrl('RESEARCH_GENERATE');

// Or for custom endpoints:
const customUrl = API_CONFIG.getApiUrl('/api/v1/custom/endpoint');
```
