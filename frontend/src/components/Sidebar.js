import React from 'react';
import { Layout, Menu } from 'antd';
import { 
  HomeOutlined, 
  SearchOutlined, 
  NodeIndexOutlined, 
  InfoCircleOutlined 
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';

const { Sider } = Layout;

function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <HomeOutlined />,
      label: 'Home',
    },
    {
      key: '/research',
      icon: <SearchOutlined />,
      label: 'Research',
    },
    {
      key: '/knowledge-graph',
      icon: <NodeIndexOutlined />,
      label: 'Knowledge Graph',
    },
    {
      key: '/about',
      icon: <InfoCircleOutlined />,
      label: 'About',
    },
  ];

  const handleMenuClick = ({ key }) => {
    navigate(key);
  };

  return (
    <Sider width={200} theme="light">
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{ height: '100%', borderRight: 0 }}
      />
    </Sider>
  );
}

export default Sidebar;
