import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import { Layout } from 'antd';
import styled from 'styled-components';

import Header from './components/Header';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import ResearchPage from './pages/ResearchPage';
import KnowledgeGraphPage from './pages/KnowledgeGraphPage';
import AboutPage from './pages/AboutPage';

const { Content } = Layout;

const AppContainer = styled(Layout)`
  min-height: 100vh;
`;

const MainContent = styled(Content)`
  margin: 24px;
  padding: 24px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

function App() {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 6,
        },
      }}
    >
      <Router>
        <AppContainer>
          <Header />
          <Layout>
            <Sidebar />
            <Layout>
              <MainContent>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/research" element={<ResearchPage />} />
                  <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />
                  <Route path="/about" element={<AboutPage />} />
                </Routes>
              </MainContent>
            </Layout>
          </Layout>
        </AppContainer>
      </Router>
    </ConfigProvider>
  );
}

export default App;
