import React from 'react';
import { Card, Row, Col, Typography, Timeline, Tag, Space } from 'antd';
import { 
  ExperimentOutlined, 
  RobotOutlined, 
  DatabaseOutlined,
  NodeIndexOutlined,
  ApiOutlined,
  SecurityScanOutlined 
} from '@ant-design/icons';
import styled from 'styled-components';

const { Title, Paragraph } = Typography;

const FeatureSection = styled.div`
  margin: 40px 0;
`;


const ArchitectureCard = styled(Card)`
  height: 100%;
  transition: transform 0.3s ease;
  
  &:hover {
    transform: translateY(-4px);
  }
`;

function AboutPage() {
  const features = [
    {
      icon: <RobotOutlined style={{ fontSize: '24px', color: '#1890ff' }} />,
      title: 'Multi-Agent AI System',
      description: 'Specialized AI agents work together like a research team, each with unique expertise in data retrieval, extraction, synthesis, and validation.'
    },
    {
      icon: <DatabaseOutlined style={{ fontSize: '24px', color: '#52c41a' }} />,
      title: 'Advanced Vector Search',
      description: 'Domain-specific embeddings using BioBERT and semantic search for precise document retrieval from scientific literature.'
    },
    {
      icon: <NodeIndexOutlined style={{ fontSize: '24px', color: '#fa8c16' }} />,
      title: 'Knowledge Graphs',
      description: 'Knowledge graphs capture complex scientific relationships and enable advanced reasoning.'
    },
    {
      icon: <ApiOutlined style={{ fontSize: '24px', color: '#eb2f96' }} />,
      title: 'Self-Correction Loop',
      description: 'Generator-Critic pattern ensures hypothesis quality through iterative validation and improvement.'
    }
  ];


  const architecture = [
    {
      title: 'Data Ingestion',
      description: 'Scientific papers and datasets are processed and stored in vector databases with metadata indexing.',
      components: ['Semantic Scholar API', 'PubMed', 'PDF Processing', 'Text Chunking']
    },
    {
      title: 'AI Processing',
      description: 'Multi-agent system processes documents through specialized AI agents for extraction and synthesis.',
      components: ['RAG Agent', 'Extraction Agent', 'Synthesis Agent', 'Critic Agent']
    },
    {
      title: 'Knowledge Graph',
      description: 'Extracted entities and relationships are stored in knowledge graph for advanced graph-based reasoning.',
      components: ['Entity Extraction', 'Relationship Mapping', 'Graph Algorithms', 'Path Discovery']
    },
    {
      title: 'Hypothesis Generation',
      description: 'LLMs generate novel hypotheses based on retrieved knowledge and graph relationships.',
      components: ['Prompt Engineering', 'Chain-of-Thought', 'Self-Correction', 'Validation']
    }
  ];

  return (
    <div>
      <Title level={2}>
        <ExperimentOutlined /> About Synthetica
      </Title>
      <Paragraph style={{ fontSize: '18px', marginBottom: '32px' }}>
        Synthetica is an AI-powered research assistant 
        that generates novel research hypotheses from scientific literature using advanced multi-agent 
        systems, knowledge graphs, and large language models.
      </Paragraph>

      <FeatureSection>
        <Title level={3}>Key Features</Title>
        <Row gutter={[24, 24]}>
          {features.map((feature, index) => (
            <Col xs={24} sm={12} key={index}>
              <Card>
                <Space direction="vertical" style={{ width: '100%' }}>
                  {feature.icon}
                  <Title level={4}>{feature.title}</Title>
                  <Paragraph>{feature.description}</Paragraph>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </FeatureSection>


      <FeatureSection>
        <Title level={3}>System Architecture</Title>
        <Row gutter={[24, 24]}>
          {architecture.map((layer, index) => (
            <Col xs={24} sm={12} lg={6} key={index}>
              <ArchitectureCard>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Title level={4}>{layer.title}</Title>
                  <Paragraph>{layer.description}</Paragraph>
                  <div>
                    <Title level={5}>Components:</Title>
                    <Space wrap>
                      {layer.components.map((component, compIndex) => (
                        <Tag key={compIndex} color="green">{component}</Tag>
                      ))}
                    </Space>
                  </div>
                </Space>
              </ArchitectureCard>
            </Col>
          ))}
        </Row>
      </FeatureSection>

      <FeatureSection>
        <Title level={3}>How It Works</Title>
        <Timeline
          items={[
            {
              children: (
                <div>
                  <Title level={5}>1. Query Processing</Title>
                  <Paragraph>User submits a research question or topic of interest.</Paragraph>
                </div>
              )
            },
            {
              children: (
                <div>
                  <Title level={5}>2. Document Retrieval</Title>
                  <Paragraph>RAG agent searches vector database for relevant scientific papers and abstracts.</Paragraph>
                </div>
              )
            },
            {
              children: (
                <div>
                  <Title level={5}>3. Entity Extraction</Title>
                  <Paragraph>BioBERT-based extraction agent identifies genes, proteins, diseases, and other biomedical entities.</Paragraph>
                </div>
              )
            },
            {
              children: (
                <div>
                  <Title level={5}>4. Knowledge Graph Building</Title>
                  <Paragraph>Extracted entities and relationships are stored in knowledge graph.</Paragraph>
                </div>
              )
            },
            {
              children: (
                <div>
                  <Title level={5}>5. Hypothesis Generation</Title>
                  <Paragraph>Synthesis agent uses LLMs to generate novel, testable research hypotheses.</Paragraph>
                </div>
              )
            },
            {
              children: (
                <div>
                  <Title level={5}>6. Validation & Refinement</Title>
                  <Paragraph>Critic agent validates hypotheses and triggers self-correction loop if needed.</Paragraph>
                </div>
              )
            }
          ]}
        />
      </FeatureSection>

      <FeatureSection>
        <Title level={3}>Research Applications</Title>
        <Row gutter={[24, 24]}>
          <Col xs={24} sm={12}>
            <Card title="Biomedical Research">
              <ul>
                <li>Drug discovery and repurposing</li>
                <li>Disease mechanism identification</li>
                <li>Biomarker discovery</li>
                <li>Pathway analysis</li>
              </ul>
            </Card>
          </Col>
          <Col xs={24} sm={12}>
            <Card title="Clinical Applications">
              <ul>
                <li>Treatment optimization</li>
                <li>Patient stratification</li>
                <li>Clinical trial design</li>
                <li>Precision medicine</li>
              </ul>
            </Card>
          </Col>
        </Row>
      </FeatureSection>

      <FeatureSection>
        <Title level={3}>Future Enhancements</Title>
        <Card>
          <Row gutter={[24, 24]}>
            <Col xs={24} sm={8}>
              <div>
                <Title level={5}>Advanced AI</Title>
                <ul>
                  <li>Graph Neural Networks</li>
                  <li>Reinforcement Learning</li>
                  <li>Multi-modal AI</li>
                </ul>
              </div>
            </Col>
            <Col xs={24} sm={8}>
              <div>
                <Title level={5}>Privacy & Security</Title>
                <ul>
                  <li>Homomorphic Encryption</li>
                  <li>Federated Learning</li>
                  <li>Secure Multi-party Computation</li>
                </ul>
              </div>
            </Col>
            <Col xs={24} sm={8}>
              <div>
                <Title level={5}>Scalability</Title>
                <ul>
                  <li>Cloud-native Architecture</li>
                  <li>Distributed Processing</li>
                  <li>Real-time Updates</li>
                </ul>
              </div>
            </Col>
          </Row>
        </Card>
      </FeatureSection>
    </div>
  );
}

export default AboutPage;
