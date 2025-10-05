import React from 'react';
import { Card, Row, Col, Typography, Button, Space, Statistic } from 'antd';
import { 
  ExperimentOutlined, 
  SearchOutlined, 
  NodeIndexOutlined,
  RocketOutlined 
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';

const { Title, Paragraph } = Typography;

const HeroSection = styled.div`
  text-align: center;
  padding: 60px 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  margin: -24px -24px 40px -24px;
  border-radius: 0 0 16px 16px;
`;

const FeatureCard = styled(Card)`
  height: 100%;
  transition: transform 0.3s ease;
  
  &:hover {
    transform: translateY(-4px);
  }
`;

const StatsSection = styled.div`
  background: #f8f9fa;
  padding: 40px 0;
  margin: 40px -24px -24px -24px;
  border-radius: 16px 16px 0 0;
`;

function HomePage() {
  const navigate = useNavigate();

  const features = [
    {
      icon: <SearchOutlined style={{ fontSize: '32px', color: '#1890ff' }} />,
      title: 'Intelligent Research',
      description: 'AI-powered document retrieval and analysis using advanced NLP and vector databases.',
    },
    {
      icon: <NodeIndexOutlined style={{ fontSize: '32px', color: '#52c41a' }} />,
      title: 'Knowledge Graphs',
      description: 'Build and explore scientific relationships using graph database.',
    },
    {
      icon: <ExperimentOutlined style={{ fontSize: '32px', color: '#fa8c16' }} />,
      title: 'Hypothesis Generation',
      description: 'Generate novel, testable research hypotheses using multi-agent AI systems.',
    },
    {
      icon: <RocketOutlined style={{ fontSize: '32px', color: '#eb2f96' }} />,
      title: 'Self-Correction',
      description: 'Advanced critic agents validate and improve generated hypotheses.',
    },
  ];

  return (
    <div>
      <HeroSection>
        <Title level={1} style={{ color: 'white', marginBottom: '16px' }}>
          Synthetica
        </Title>
        <Title level={3} style={{ color: 'white', fontWeight: 'normal', marginBottom: '24px' }}>
          AI-Powered Scientific Research Assistant
        </Title>
        <Paragraph style={{ color: 'white', fontSize: '18px', marginBottom: '32px' }}>
          An AI-powered scientific research assistant that generates novel research hypotheses 
          from biomedical literature using multi-agent systems and knowledge graphs.
        </Paragraph>
        <Space size="large">
          <Button 
            type="primary" 
            size="large"
            onClick={() => navigate('/research')}
          >
            Start Research
          </Button>
          <Button 
            size="large"
            onClick={() => navigate('/about')}
          >
            Learn More
          </Button>
        </Space>
      </HeroSection>

      <Row gutter={[24, 24]}>
        {features.map((feature, index) => (
          <Col xs={24} sm={12} lg={6} key={index}>
            <FeatureCard>
              <div style={{ textAlign: 'center', padding: '20px 0' }}>
                {feature.icon}
                <Title level={4} style={{ marginTop: '16px' }}>
                  {feature.title}
                </Title>
                <Paragraph style={{ color: '#666' }}>
                  {feature.description}
                </Paragraph>
              </div>
            </FeatureCard>
          </Col>
        ))}
      </Row>

      <StatsSection>
        <Row gutter={[24, 24]}>
          <Col xs={24} sm={6}>
            <Statistic
              title="Research Domains"
              value={6}
              suffix="+"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col xs={24} sm={6}>
            <Statistic
              title="AI Agents"
              value={4}
              suffix="+"
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col xs={24} sm={6}>
            <Statistic
              title="Knowledge Graph Nodes"
              value={10000}
              suffix="+"
              valueStyle={{ color: '#fa8c16' }}
            />
          </Col>
          <Col xs={24} sm={6}>
            <Statistic
              title="Processing Speed"
              value={5}
              suffix="x faster"
              valueStyle={{ color: '#eb2f96' }}
            />
          </Col>
        </Row>
      </StatsSection>
    </div>
  );
}

export default HomePage;
