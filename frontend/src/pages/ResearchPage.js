import React, { useState } from 'react';
import { 
  Card, 
  Form, 
  Input, 
  Select, 
  Button, 
  Space, 
  Typography, 
  Alert, 
  Spin,
  List,
  Tag,
  Progress,
  Divider
} from 'antd';
import { 
  SearchOutlined, 
  ExperimentOutlined, 
  CheckCircleOutlined,
  ClockCircleOutlined 
} from '@ant-design/icons';
import styled from 'styled-components';
import axios from 'axios';

const { Title, Paragraph, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

const ResearchForm = styled(Card)`
  margin-bottom: 24px;
`;

const ResultsSection = styled.div`
  margin-top: 24px;
`;

const HypothesisCard = styled(Card)`
  margin-bottom: 16px;
  border-left: 4px solid #1890ff;
`;

const StatusTag = styled(Tag)`
  margin-left: 8px;
`;

function ResearchPage() {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [workflowId, setWorkflowId] = useState(null);
  const [workflowStatus, setWorkflowStatus] = useState(null);

  const domains = [
    { value: 'biomedical', label: 'Biomedical' },
    { value: 'clinical', label: 'Clinical' },
    { value: 'pharmaceutical', label: 'Pharmaceutical' },
    { value: 'genetics', label: 'Genetics' },
    { value: 'neuroscience', label: 'Neuroscience' },
    { value: 'oncology', label: 'Oncology' },
  ];

  const handleSubmit = async (values) => {
    console.log('Form submitted with values:', values);
    setLoading(true);
    setResults(null);
    setWorkflowId(null);
    setWorkflowStatus(null);

    try {
      console.log('Sending request to backend...');
      const response = await axios.post('http://localhost:8001/api/v1/research/generate', {
        query: values.query,
        domain: values.domain,
        max_hypotheses: values.max_hypotheses,
        include_sources: true,
        depth: values.depth || 2
      });

      console.log('Response received:', response.data);
      setResults(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Research request failed:', error);
      setLoading(false);
    }
  };

  const handleAsyncSubmit = async (values) => {
    setLoading(true);
    setResults(null);
    setWorkflowStatus(null);

    try {
          const response = await axios.post('http://localhost:8001/api/v1/research/async', {
        query: values.query,
        domain: values.domain,
        max_hypotheses: values.max_hypotheses,
        include_sources: true,
        depth: values.depth || 2
      });

      setWorkflowId(response.data.workflow_id);
      setLoading(false);
      
      // Start polling for status
      pollWorkflowStatus(response.data.workflow_id);
    } catch (error) {
      console.error('Async research request failed:', error);
      setLoading(false);
    }
  };

  const pollWorkflowStatus = async (id) => {
    try {
          const response = await axios.get(`http://localhost:8001/api/v1/research/status/${id}`);
      setWorkflowStatus(response.data);
      
      if (response.data.status === 'running') {
        setTimeout(() => pollWorkflowStatus(id), 2000);
      } else if (response.data.status === 'completed') {
        // Fetch results
            const resultResponse = await axios.get(`http://localhost:8001/api/v1/research/result/${id}`);
        setResults(resultResponse.data);
      }
    } catch (error) {
      console.error('Failed to poll workflow status:', error);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'PASS': return 'success';
      case 'FAIL': return 'error';
      case 'NEEDS_REVISION': return 'warning';
      default: return 'default';
    }
  };

  return (
    <div>
      <Title level={2}>
        <SearchOutlined /> Research Hypothesis Generation
      </Title>
      <Paragraph>
        Generate novel research hypotheses using AI-powered analysis of scientific literature.
      </Paragraph>

      <ResearchForm title="Research Query">
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
        >
          <Form.Item
            name="query"
            label="Research Question"
            rules={[{ required: true, message: 'Please enter your research question' }]}
          >
            <TextArea
              rows={4}
              placeholder="Enter your research question or topic of interest..."
            />
          </Form.Item>

          <Form.Item
            name="domain"
            label="Research Domain"
            initialValue="biomedical"
            rules={[{ required: true }]}
          >
            <Select placeholder="Select research domain">
              {domains.map(domain => (
                <Option key={domain.value} value={domain.value}>
                  {domain.label}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="max_hypotheses"
            label="Maximum Hypotheses"
            initialValue={5}
          >
            <Select>
              <Option value={3}>3</Option>
              <Option value={5}>5</Option>
              <Option value={10}>10</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="depth"
            label="Research Depth"
            initialValue={2}
          >
            <Select>
              <Option value={1}>Basic (1)</Option>
              <Option value={2}>Standard (2)</Option>
              <Option value={3}>Deep (3)</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                icon={<SearchOutlined />}
              >
                Generate Hypotheses
              </Button>
              <Button
                onClick={() => form.submit()}
                loading={loading}
                icon={<ClockCircleOutlined />}
              >
                Generate Async
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </ResearchForm>

      {workflowStatus && (
        <Card title="Workflow Status" style={{ marginBottom: 24 }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Status: </Text>
              <Tag color={workflowStatus.status === 'completed' ? 'success' : 'processing'}>
                {workflowStatus.status}
              </Tag>
            </div>
            <div>
              <Text strong>Current Step: </Text>
              <Text>{workflowStatus.current_step}</Text>
            </div>
            <Progress percent={Math.round(workflowStatus.progress * 100)} />
          </Space>
        </Card>
      )}

      {loading && (
        <Card>
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Spin size="large" />
            <div style={{ marginTop: '16px' }}>
              <Text>Generating hypotheses...</Text>
            </div>
          </div>
        </Card>
      )}

      {results && (
        <ResultsSection>
          <Card title="Research Results">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>Query: </Text>
                <Text>{results.query}</Text>
              </div>
              <div>
                <Text strong>Domain: </Text>
                <Tag>{results.domain}</Tag>
              </div>
              <div>
                <Text strong>Processing Time: </Text>
                <Text>{results.processing_time.toFixed(2)}s</Text>
              </div>
              <div>
                <Text strong>Documents Processed: </Text>
                <Text>{results.total_documents}</Text>
              </div>
            </Space>
          </Card>

          <Title level={3} style={{ marginTop: '24px' }}>
            <ExperimentOutlined /> Generated Hypotheses
          </Title>

          {results.hypotheses.map((hypothesis, index) => (
            <HypothesisCard key={index} title={hypothesis.title}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>Description: </Text>
                  <Paragraph>{hypothesis.description}</Paragraph>
                </div>
                
                <div>
                  <Text strong>Rationale: </Text>
                  <Paragraph>{hypothesis.rationale}</Paragraph>
                </div>

                {hypothesis.testable_predictions.length > 0 && (
                  <div>
                    <Text strong>Testable Predictions: </Text>
                    <List
                      size="small"
                      dataSource={hypothesis.testable_predictions}
                      renderItem={item => <List.Item>{item}</List.Item>}
                    />
                  </div>
                )}

                {hypothesis.methodology.length > 0 && (
                  <div>
                    <Text strong>Suggested Methodology: </Text>
                    <List
                      size="small"
                      dataSource={hypothesis.methodology}
                      renderItem={item => <List.Item>{item}</List.Item>}
                    />
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Text strong>Confidence: </Text>
                    <Text>{(hypothesis.confidence_score * 100).toFixed(1)}%</Text>
                  </div>
                  <div>
                    <Text strong>Status: </Text>
                    <StatusTag color={getStatusColor(hypothesis.status)}>
                      {hypothesis.status}
                    </StatusTag>
                  </div>
                </div>
              </Space>
            </HypothesisCard>
          ))}
        </ResultsSection>
      )}
    </div>
  );
}

export default ResearchPage;
