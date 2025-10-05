import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  Input, 
  Button, 
  Space, 
  Typography, 
  Alert, 
  Spin,
  List,
  Tag,
  Row,
  Col,
  Statistic
} from 'antd';
import { 
  NodeIndexOutlined, 
  SearchOutlined,
  LinkOutlined,
  DatabaseOutlined 
} from '@ant-design/icons';
import styled from 'styled-components';
import axios from 'axios';

const { Title, Paragraph, Text } = Typography;

const GraphContainer = styled.div`
  margin: 24px 0;
`;

const SearchSection = styled(Card)`
  margin-bottom: 24px;
`;

const StatsSection = styled(Card)`
  margin-bottom: 24px;
`;

const ResultsSection = styled.div`
  margin-top: 24px;
`;

function KnowledgeGraphPage() {
  const [loading, setLoading] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [pathResults, setPathResults] = useState(null);
  const [graphStats, setGraphStats] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [entity1, setEntity1] = useState('');
  const [entity2, setEntity2] = useState('');

  useEffect(() => {
    // Load initial graph statistics
    loadGraphStats();
  }, []);

  const loadGraphStats = async () => {
    try {
      // This would be a real API call in production
      setGraphStats({
        total_nodes: 12543,
        total_relationships: 45678,
        node_types: {
          'Gene': 5432,
          'Protein': 3210,
          'Disease': 1890,
          'Compound': 2011
        }
      });
    } catch (error) {
      console.error('Failed to load graph stats:', error);
    }
  };

  const handleEntitySearch = async () => {
    if (!searchTerm.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.get(
        `http://localhost:8000/api/v1/research/entities/${encodeURIComponent(searchTerm)}`
      );
      setSearchResults(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Entity search failed:', error);
      setLoading(false);
    }
  };

  const handlePathSearch = async () => {
    if (!entity1.trim() || !entity2.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.get(
        `http://localhost:8000/api/v1/research/paths/${encodeURIComponent(entity1)}/${encodeURIComponent(entity2)}`
      );
      setPathResults(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Path search failed:', error);
      setLoading(false);
    }
  };

  const getNodeTypeColor = (type) => {
    const colors = {
      'Gene': '#1890ff',
      'Protein': '#52c41a',
      'Disease': '#fa8c16',
      'Compound': '#eb2f96',
      'Pathway': '#722ed1',
      'default': '#666'
    };
    return colors[type] || colors.default;
  };

  return (
    <div>
      <Title level={2}>
        <NodeIndexOutlined /> Knowledge Graph Explorer
      </Title>
      <Paragraph>
        Explore scientific relationships and discover connections between entities in the knowledge graph.
      </Paragraph>

      {graphStats && (
        <StatsSection title="Graph Statistics">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={6}>
              <Statistic
                title="Total Nodes"
                value={graphStats.total_nodes}
                prefix={<DatabaseOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="Total Relationships"
                value={graphStats.total_relationships}
                prefix={<LinkOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="Node Types"
                value={Object.keys(graphStats.node_types).length}
                suffix="+"
                valueStyle={{ color: '#fa8c16' }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="Density"
                value={((graphStats.total_relationships / graphStats.total_nodes) * 100).toFixed(1)}
                suffix="%"
                valueStyle={{ color: '#eb2f96' }}
              />
            </Col>
          </Row>
        </StatsSection>
      )}

      <SearchSection title="Entity Search">
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Search for related entities:</Text>
          </div>
          <Space.Compact style={{ width: '100%' }}>
            <Input
              placeholder="Enter entity name (e.g., 'BRCA1', 'cancer', 'insulin')"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onPressEnter={handleEntitySearch}
            />
            <Button
              type="primary"
              icon={<SearchOutlined />}
              onClick={handleEntitySearch}
              loading={loading}
            >
              Search
            </Button>
          </Space.Compact>
        </Space>
      </SearchSection>

      <SearchSection title="Path Discovery">
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Find paths between two entities:</Text>
          </div>
          <Row gutter={16}>
            <Col xs={24} sm={10}>
              <Input
                placeholder="Entity 1"
                value={entity1}
                onChange={(e) => setEntity1(e.target.value)}
              />
            </Col>
            <Col xs={24} sm={4} style={{ textAlign: 'center' }}>
              <Text>to</Text>
            </Col>
            <Col xs={24} sm={10}>
              <Input
                placeholder="Entity 2"
                value={entity2}
                onChange={(e) => setEntity2(e.target.value)}
              />
            </Col>
          </Row>
          <Button
            type="primary"
            icon={<LinkOutlined />}
            onClick={handlePathSearch}
            loading={loading}
            disabled={!entity1.trim() || !entity2.trim()}
          >
            Find Paths
          </Button>
        </Space>
      </SearchSection>

      {loading && (
        <Card>
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Spin size="large" />
            <div style={{ marginTop: '16px' }}>
              <Text>Searching knowledge graph...</Text>
            </div>
          </div>
        </Card>
      )}

      {searchResults && (
        <ResultsSection>
          <Card title={`Related Entities for "${searchResults.entity_name}"`}>
            <List
              dataSource={searchResults.related_entities}
              renderItem={(item, index) => (
                <List.Item>
                  <Space>
                    <Tag color={getNodeTypeColor(item.labels?.[0])}>
                      {item.labels?.[0] || 'Entity'}
                    </Tag>
                    <Text strong>{item.name}</Text>
                    <Text type="secondary">Distance: {item.distance}</Text>
                  </Space>
                </List.Item>
              )}
            />
            <div style={{ marginTop: '16px' }}>
              <Text strong>Total found: {searchResults.total_found}</Text>
            </div>
          </Card>
        </ResultsSection>
      )}

      {pathResults && (
        <ResultsSection>
          <Card title={`Paths between "${pathResults.entity1}" and "${pathResults.entity2}"`}>
            {pathResults.paths.length > 0 ? (
              <List
                dataSource={pathResults.paths}
                renderItem={(path, index) => (
                  <List.Item>
                    <Card size="small" style={{ width: '100%' }}>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text strong>Path {index + 1} (Length: {path.length})</Text>
                        </div>
                        <div>
                          {path.nodes.map((node, nodeIndex) => (
                            <React.Fragment key={nodeIndex}>
                              <Tag color={getNodeTypeColor(node.labels?.[0])}>
                                {node.name}
                              </Tag>
                              {nodeIndex < path.nodes.length - 1 && (
                                <Text type="secondary"> → </Text>
                              )}
                            </React.Fragment>
                          ))}
                        </div>
                        {path.relationships.length > 0 && (
                          <div>
                            <Text strong>Relationships: </Text>
                            {path.relationships.map((rel, relIndex) => (
                              <Tag key={relIndex} color="blue">
                                {rel.type}
                              </Tag>
                            ))}
                          </div>
                        )}
                      </Space>
                    </Card>
                  </List.Item>
                )}
              />
            ) : (
              <Alert
                message="No paths found"
                description="No direct or indirect paths were found between these entities."
                type="info"
              />
            )}
            <div style={{ marginTop: '16px' }}>
              <Text strong>Total paths found: {pathResults.total_paths}</Text>
            </div>
          </Card>
        </ResultsSection>
      )}

      <GraphContainer>
        <Card title="Interactive Graph Visualization">
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <NodeIndexOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
            <div style={{ marginTop: '16px' }}>
              <Text>Interactive graph visualization will be implemented here</Text>
              <br />
              <Text type="secondary">
                This will use Cytoscape.js to display the knowledge graph interactively
              </Text>
            </div>
          </div>
        </Card>
      </GraphContainer>
    </div>
  );
}

export default KnowledgeGraphPage;
