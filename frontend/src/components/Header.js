import React from 'react';
import { Layout, Typography, Space, Button } from 'antd';
import { ExperimentOutlined, GithubOutlined } from '@ant-design/icons';
import styled from 'styled-components';

const { Header: AntHeader } = Layout;
const { Title } = Typography;

const StyledHeader = styled(AntHeader)`
  background: #fff;
  padding: 0 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const HeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
`;

function Header() {
  return (
    <StyledHeader>
      <Logo>
        <ExperimentOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
        <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
          Synthetica
        </Title>
        <span style={{ color: '#666', fontSize: '14px' }}>
          Scientific Research Assistant
        </span>
      </Logo>
      
      <HeaderActions>
        <Button 
          type="text" 
          icon={<GithubOutlined />}
          href="https://github.com/your-repo/synthetica"
          target="_blank"
        >
          GitHub
        </Button>
      </HeaderActions>
    </StyledHeader>
  );
}

export default Header;
