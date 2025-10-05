// API Configuration
const API_CONFIG = {
  // Base URLs
  BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  BASE_URL_CONTAINER: process.env.REACT_APP_API_BASE_URL_CONTAINER || 'http://synthetica-backend-full:8000',
  
  // API Endpoints
  ENDPOINTS: {
    RESEARCH_GENERATE: process.env.REACT_APP_API_RESEARCH_GENERATE || '/api/v1/research/generate',
    RESEARCH_ASYNC: process.env.REACT_APP_API_RESEARCH_ASYNC || '/api/v1/research/async',
    RESEARCH_STATUS: process.env.REACT_APP_API_RESEARCH_STATUS || '/api/v1/research/status',
    RESEARCH_RESULT: process.env.REACT_APP_API_RESEARCH_RESULT || '/api/v1/research/result',
    RESEARCH_ENTITIES: process.env.REACT_APP_API_RESEARCH_ENTITIES || '/api/v1/research/entities',
    RESEARCH_PATHS: process.env.REACT_APP_API_RESEARCH_PATHS || '/api/v1/research/paths',
  },
  
  // Environment
  ENVIRONMENT: process.env.REACT_APP_ENVIRONMENT || 'development',
  
  // Helper functions
  getApiUrl: (endpoint) => {
    const baseUrl = API_CONFIG.ENVIRONMENT === 'production' 
      ? API_CONFIG.BASE_URL_CONTAINER 
      : API_CONFIG.BASE_URL;
    return `${baseUrl}${endpoint}`;
  },
  
  getFullUrl: (endpoint) => {
    return API_CONFIG.getApiUrl(API_CONFIG.ENDPOINTS[endpoint]);
  }
};

export default API_CONFIG;
