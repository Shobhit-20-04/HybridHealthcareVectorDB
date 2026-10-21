import React, { useState, useEffect } from 'react';
import {
  Database,
  Search,
  Image as ImageIcon,
  Shield,
  Bot,
  Activity,
  Layers,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Filter,
  FileText,
  User,
  HeartPulse,
  ChevronRight,
  ExternalLink,
  Code
} from 'lucide-react';

const PRESET_QUERIES = [
  "progressive dyspnea on exertion and pulmonary infiltrate",
  "acute myocardial infarction with ST elevation",
  "sarcoidosis and lymphadenopathy",
  "uncontrolled hyperglycemia and insulin resistance"
];

const PRESET_RAG_QUESTIONS = [
  { label: "Grounded Medical Case", text: "What evidence exists regarding sarcoidosis and myocardial infarction?" },
  { label: "Abstention Test (Off-Corpus)", text: "What surgical trephine diameter was used for pediatric craniotomy?" }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('hybrid');
  const [systemStats, setSystemStats] = useState(null);
  const [loadingStats, setLoadingStats] = useState(true);

  // Hybrid Search State
  const [hybridQuery, setHybridQuery] = useState(PRESET_QUERIES[0]);
  const [deptFilter, setDeptFilter] = useState('ALL');
  const [docTypeFilter, setDocTypeFilter] = useState('ALL');
  const [minAge, setMinAge] = useState('');
  const [maxAge, setMaxAge] = useState('');
  const [searchMode, setSearchMode] = useState('hnsw');
  const [topK, setTopK] = useState(5);
  const [showExplain, setShowExplain] = useState(true);
  const [hybridResults, setHybridResults] = useState(null);
  const [searchingHybrid, setSearchingHybrid] = useState(false);

  // Multimodal Search State
  const [mmMode, setMmMode] = useState('text_to_image');
  const [mmTextQuery, setMmTextQuery] = useState('computed tomography axial slice chest');
  const [mmModality, setMmModality] = useState('ALL');
  const [mmBodySite, setMmBodySite] = useState('ALL');
  const [mmResults, setMmResults] = useState(null);
  const [searchingMM, setSearchingMM] = useState(false);

  // Security Simulation State
  const [secRole, setSecRole] = useState('attending_physician');
  const [secDept, setSecDept] = useState('Cardiology');
  const [secReport, setSecReport] = useState(null);
  const [testingSec, setTestingSec] = useState(false);

  // RAG Demo State
  const [ragQuestion, setRagQuestion] = useState(PRESET_RAG_QUESTIONS[0].text);
  const [ragDept, setRagDept] = useState('ALL');
  const [ragResult, setRagResult] = useState(null);
  const [runningRAG, setRunningRAG] = useState(false);

  // Load Database Metadata on mount
  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    setLoadingStats(true);
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = await res.json();
        setSystemStats(data);
      }
    } catch (err) {
      console.error("Failed to fetch database stats:", err);
    } finally {
      setLoadingStats(false);
    }
  };

  const handleHybridSearch = async (e) => {
    if (e) e.preventDefault();
    if (!hybridQuery.trim()) return;
    setSearchingHybrid(true);
    setHybridResults(null);
    try {
      const res = await fetch('/api/search/hybrid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: hybridQuery,
          top_k: topK,
          department: deptFilter,
          document_type: docTypeFilter,
          min_age: minAge ? parseInt(minAge, 10) : null,
          max_age: maxAge ? parseInt(maxAge, 10) : null,
          search_mode: searchMode,
          explain: showExplain
        })
      });
      const data = await res.json();
      setHybridResults(data);
    } catch (err) {
      console.error("Hybrid search failed:", err);
    } finally {
      setSearchingHybrid(false);
    }
  };

  const handleMultimodalSearch = async (e) => {
    if (e) e.preventDefault();
    setSearchingMM(true);
    setMmResults(null);
    try {
      const res = await fetch('/api/search/multimodal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: mmMode,
          query_text: mmMode === 'text_to_image' ? mmTextQuery : null,
          query_image_path: mmMode === 'image_to_image' ? 'data/real_pmc/images/ct_0001.png' : null,
          modality: mmModality,
          body_site: mmBodySite,
          top_k: 4
        })
      });
      const data = await res.json();
      setMmResults(data);
    } catch (err) {
      console.error("Multimodal search failed:", err);
    } finally {
      setSearchingMM(false);
    }
  };

  const handleSecurityTest = async () => {
    setTestingSec(true);
    setSecReport(null);
    try {
      const res = await fetch('/api/security/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role: secRole,
          department: secDept
        })
      });
      const data = await res.json();
      setSecReport(data);
    } catch (err) {
      console.error("Security test failed:", err);
    } finally {
      setTestingSec(false);
    }
  };

  const handleRAGExecute = async (e) => {
    if (e) e.preventDefault();
    if (!ragQuestion.trim()) return;
    setRunningRAG(true);
    setRagResult(null);
    try {
      const res = await fetch('/api/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: ragQuestion,
          department: ragDept,
          top_k: 3
        })
      });
      const data = await res.json();
      setRagResult(data);
    } catch (err) {
      console.error("RAG execution failed:", err);
    } finally {
      setRunningRAG(false);
    }
  };

  return (
    <div className="app-container">
      {/* Top Navigation & Status Bar */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="brand-icon">
              <Database size={22} />
            </div>
            <div>
              <h1 className="brand-title">Hybrid Healthcare VectorDB</h1>
              <p className="brand-subtitle">
                PostgreSQL 18.6 + pgvector v0.8.6 • Academic Relational–Vector Engine
              </p>
            </div>
          </div>
          <div className="header-meta">
            <span className="badge badge-success">
              <CheckCircle2 size={13} />
              Connected: localhost:5432
            </span>
            <span className="badge badge-primary">
              <Layers size={13} />
              17,076 Database Rows
            </span>
            <span className="badge badge-indigo">
              <HeartPulse size={13} />
              1,000 Real PMC Patients
            </span>
          </div>
        </div>
      </header>

      {/* Main Tab Navigation */}
      <nav className="nav-tabs" aria-label="Database Functionality Tabs">
        <div className="nav-tabs-inner">
          <button
            className={`nav-tab ${activeTab === 'hybrid' ? 'active' : ''}`}
            onClick={() => setActiveTab('hybrid')}
          >
            <Search size={16} />
            Hybrid Relational–Vector Search
          </button>
          <button
            className={`nav-tab ${activeTab === 'multimodal' ? 'active' : ''}`}
            onClick={() => setActiveTab('multimodal')}
          >
            <ImageIcon size={16} />
            Multimodal Imaging Search
          </button>
          <button
            className={`nav-tab ${activeTab === 'security' ? 'active' : ''}`}
            onClick={() => setActiveTab('security')}
          >
            <Shield size={16} />
            Row-Level Security (RLS) Simulator
          </button>
          <button
            className={`nav-tab ${activeTab === 'rag' ? 'active' : ''}`}
            onClick={() => setActiveTab('rag')}
          >
            <Bot size={16} />
            Clinical RAG Assistant
          </button>
          <button
            className={`nav-tab ${activeTab === 'telemetry' ? 'active' : ''}`}
            onClick={() => setActiveTab('telemetry')}
          >
            <Activity size={16} />
            Database Architecture & Telemetry
          </button>
        </div>
      </nav>

      {/* Main Workspace */}
      <main className="app-main">
        {/* ================================================================= */}
        {/* TAB 1: HYBRID RELATIONAL-VECTOR SEARCH                            */}
        {/* ================================================================= */}
        {activeTab === 'hybrid' && (
          <section aria-labelledby="hybrid-search-title">
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 id="hybrid-search-title" className="card-title">
                    <Search size={18} className="text-primary" />
                    Hybrid Relational–Vector Query Engine
                  </h2>
                  <p className="card-desc">
                    Combines relational SQL predicates (Department, Age, Document Type) with high-dimensional 384-d dense vector cosine distance ranking in a single transaction.
                  </p>
                </div>
              </div>

              <form onSubmit={handleHybridSearch}>
                <div className="form-group">
                  <label className="form-label" htmlFor="hybrid-query-input">
                    Clinical Query (Natural Language Symptoms / Findings)
                  </label>
                  <div className="search-input-group">
                    <input
                      id="hybrid-query-input"
                      type="text"
                      className="form-control search-input"
                      value={hybridQuery}
                      onChange={(e) => setHybridQuery(e.target.value)}
                      placeholder="e.g. progressive dyspnea on exertion with bilateral infiltrates..."
                    />
                    <button type="submit" className="btn btn-primary" disabled={searchingHybrid}>
                      {searchingHybrid ? <span className="spinner" /> : <Search size={16} />}
                      Execute Search
                    </button>
                  </div>

                  {/* Preset Query Chips */}
                  <div className="preset-queries">
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', alignSelf: 'center' }}>
                      Try Preset:
                    </span>
                    {PRESET_QUERIES.map((q, i) => (
                      <button
                        key={i}
                        type="button"
                        className="preset-chip"
                        onClick={() => {
                          setHybridQuery(q);
                        }}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Relational Predicates Grid */}
                <div className="filters-grid">
                  <div className="form-group">
                    <label className="form-label">Clinical Department</label>
                    <select
                      className="form-select"
                      value={deptFilter}
                      onChange={(e) => setDeptFilter(e.target.value)}
                    >
                      <option value="ALL">All Departments (Unfiltered)</option>
                      <option value="Cardiology">Cardiology</option>
                      <option value="Pulmonology">Pulmonology</option>
                      <option value="Neurology">Neurology</option>
                      <option value="Oncology">Oncology</option>
                      <option value="Emergency">Emergency</option>
                      <option value="Nephrology">Nephrology</option>
                      <option value="Gastroenterology">Gastroenterology</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Document Type</label>
                    <select
                      className="form-select"
                      value={docTypeFilter}
                      onChange={(e) => setDocTypeFilter(e.target.value)}
                    >
                      <option value="ALL">All Document Types</option>
                      <option value="progress_note">Progress Note</option>
                      <option value="discharge_summary">Discharge Summary</option>
                      <option value="radiology_report">Radiology Report</option>
                      <option value="consult_note">Consult Note</option>
                      <option value="pathology_report">Pathology Report</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Min Patient Age</label>
                    <input
                      type="number"
                      className="form-control"
                      placeholder="e.g. 40"
                      value={minAge}
                      onChange={(e) => setMinAge(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Max Patient Age</label>
                    <input
                      type="number"
                      className="form-control"
                      placeholder="e.g. 80"
                      value={maxAge}
                      onChange={(e) => setMaxAge(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Vector Search Mode</label>
                    <select
                      className="form-select"
                      value={searchMode}
                      onChange={(e) => setSearchMode(e.target.value)}
                    >
                      <option value="hnsw">HNSW (Navigable Proximity Graph)</option>
                      <option value="exact_flat">Exact Flat (Sequential Scan Baseline)</option>
                      <option value="ivfflat">IVFFlat (Inverted Centroids)</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Top-K Nearest</label>
                    <select
                      className="form-select"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                    >
                      <option value="3">Top 3 Hits</option>
                      <option value="5">Top 5 Hits</option>
                      <option value="10">Top 10 Hits</option>
                    </select>
                  </div>
                </div>

                <div style={{ marginTop: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="checkbox"
                    id="explain-checkbox"
                    checked={showExplain}
                    onChange={(e) => setShowExplain(e.target.checked)}
                  />
                  <label htmlFor="explain-checkbox" style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)', cursor: 'pointer' }}>
                    Inspect PostgreSQL Execution Plan (EXPLAIN ANALYZE BUFFERS)
                  </label>
                </div>
              </form>
            </div>

            {/* Execution Telemetry Bar */}
            {hybridResults && (
              <div className="card" style={{ padding: '0.85rem 1.25rem', backgroundColor: '#f8fafc' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Clock size={16} className="text-muted" />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Execution Latency:</span>
                    <span className="metric-pill green">{hybridResults.timing_ms?.total} ms total</span>
                    <span className="metric-pill blue">DB: {hybridResults.timing_ms?.db_query} ms</span>
                    <span className="metric-pill gray">Embed: {hybridResults.timing_ms?.embed} ms</span>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                      Returned <strong>{hybridResults.returned_count}</strong> records across 6,244 real PMC chunks
                    </span>
                  </div>
                </div>

                {/* Plan Accordion */}
                {hybridResults.plan && (
                  <details style={{ marginTop: '0.75rem' }}>
                    <summary style={{ cursor: 'pointer', fontSize: '0.82rem', fontWeight: 600, color: 'var(--primary)' }}>
                      View PostgreSQL Physical Execution Plan & Optimizer Strategy
                    </summary>
                    <pre className="code-block">{hybridResults.plan}</pre>
                  </details>
                )}
              </div>
            )}

            {/* Results Stream */}
            {hybridResults?.results?.map((item) => (
              <div key={item.chunk_id} className="result-card">
                <div className="result-header">
                  <div>
                    <h3 className="result-title">
                      #{item.rank} {item.document_title}
                    </h3>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                      Doc ID: <strong style={{ color: 'var(--text-main)' }}>{item.synthetic_document_num}</strong> • Type:{' '}
                      <span className="badge badge-subtle">{item.document_type}</span>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span className="metric-pill blue" style={{ fontSize: '0.85rem' }}>
                      Distance: {item.cosine_distance.toFixed(4)}
                    </span>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                      Similarity: {(item.cosine_similarity * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="result-meta-row">
                  <span>
                    <User size={13} style={{ display: 'inline', verticalAlign: '-2px', marginRight: '3px' }} />
                    MRN: <strong>{item.synthetic_mrn}</strong> ({item.patient_age}y {item.gender})
                  </span>
                  <span>•</span>
                  <span>
                    Dept: <strong>{item.department}</strong>
                  </span>
                  <span>•</span>
                  <span>Type: {item.encounter_type}</span>
                  <span>•</span>
                  <span>
                    ICD-10:{' '}
                    {item.diagnoses?.map((d, idx) => (
                      <span key={idx} className="badge badge-indigo" style={{ marginLeft: '4px' }}>
                        {d.code}: {d.description}
                      </span>
                    ))}
                  </span>
                </div>

                <div className="result-passage">
                  <p>"{item.snippet}"</p>
                </div>
              </div>
            ))}

            {hybridResults && hybridResults.returned_count === 0 && (
              <div className="card" style={{ textAlign: 'center', padding: '2.5rem' }}>
                <AlertTriangle size={32} style={{ color: 'var(--warning)', margin: '0 auto 0.75rem' }} />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Zero Qualifying Documents Retrieved</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '480px', margin: '0.35rem auto 0' }}>
                  The active relational filters (Department: {deptFilter}, Age Range: {minAge || '0'}-{maxAge || '120'}) are highly restrictive. No records satisfied all relational predicates and vector threshold simultaneously.
                </p>
              </div>
            )}
          </section>
        )}

        {/* ================================================================= */}
        {/* TAB 2: MULTIMODAL IMAGING SEARCH                                  */}
        {/* ================================================================= */}
        {activeTab === 'multimodal' && (
          <section aria-labelledby="multimodal-title">
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 id="multimodal-title" className="card-title">
                    <ImageIcon size={18} className="text-primary" />
                    Multimodal Diagnostic Imaging Retrieval
                  </h2>
                  <p className="card-desc">
                    Retrieves radiological examinations (CT, Digital Radiography, Ultrasound) across a unified 512-dimensional CLIP latent space with strict modality separation from text embeddings.
                  </p>
                </div>
              </div>

              <form onSubmit={handleMultimodalSearch}>
                <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
                  <button
                    type="button"
                    className={`btn ${mmMode === 'text_to_image' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setMmMode('text_to_image')}
                  >
                    Cross-Modal: Text-to-Image
                  </button>
                  <button
                    type="button"
                    className={`btn ${mmMode === 'image_to_image' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setMmMode('image_to_image')}
                  >
                    Visual: Image-to-Image
                  </button>
                </div>

                {mmMode === 'text_to_image' ? (
                  <div className="form-group">
                    <label className="form-label">Clinical Imaging Query</label>
                    <div className="search-input-group">
                      <input
                        type="text"
                        className="form-control search-input"
                        value={mmTextQuery}
                        onChange={(e) => setMmTextQuery(e.target.value)}
                        placeholder="e.g. computed tomography axial slice chest..."
                      />
                      <button type="submit" className="btn btn-primary" disabled={searchingMM}>
                        {searchingMM ? <span className="spinner" /> : <Search size={16} />}
                        Search Images
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="form-group">
                    <label className="form-label">Query Physical Image Sample</label>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                      <span className="badge badge-subtle">Target: data/real_pmc/images/ct_0001.png (CT Brain)</span>
                      <button type="submit" className="btn btn-primary" disabled={searchingMM}>
                        {searchingMM ? <span className="spinner" /> : <Search size={16} />}
                        Find Similar Images
                      </button>
                    </div>
                  </div>
                )}

                <div className="filters-grid">
                  <div className="form-group">
                    <label className="form-label">Modality Filter</label>
                    <select
                      className="form-select"
                      value={mmModality}
                      onChange={(e) => setMmModality(e.target.value)}
                    >
                      <option value="ALL">All Modalities</option>
                      <option value="CT">CT (Computed Tomography)</option>
                      <option value="DX">DX (Digital Radiography)</option>
                      <option value="US">US (Ultrasound)</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">Anatomical Body Site</label>
                    <select
                      className="form-select"
                      value={mmBodySite}
                      onChange={(e) => setMmBodySite(e.target.value)}
                    >
                      <option value="ALL">All Body Sites</option>
                      <option value="Chest">Chest</option>
                      <option value="Brain">Brain</option>
                      <option value="Abdomen">Abdomen</option>
                    </select>
                  </div>
                </div>
              </form>
            </div>

            {/* Image Results Grid */}
            {mmResults && (
              <div className="image-grid">
                {mmResults.results?.map((img) => {
                  const filename = img.synthetic_file_path?.split('/')?.pop() || '';
                  return (
                    <div key={img.image_id} className="image-card">
                      <div className="image-preview-container">
                        <img
                          src={`/api/images/${filename}`}
                          alt={`${img.modality} examination`}
                          className="image-preview"
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                      <div className="image-details">
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span className="badge badge-primary">{img.modality}</span>
                            <span className="metric-pill blue">{img.cosine_distance.toFixed(4)} dist</span>
                          </div>
                          <div style={{ marginTop: '0.45rem', fontSize: '0.85rem', fontWeight: 700 }}>
                            {img.body_site} Examination ({img.synthetic_image_num})
                          </div>
                          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                            Patient: {img.synthetic_mrn} ({img.patient_age}y) • {img.department}
                          </div>
                          <div style={{ fontSize: '0.76rem', color: '#475569', marginTop: '0.35rem', fontStyle: 'italic' }}>
                            Report: {img.associated_report_title || 'Radiology Case Report'}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}

        {/* ================================================================= */}
        {/* TAB 3: ROW-LEVEL SECURITY (RLS) SIMULATOR                         */}
        {/* ================================================================= */}
        {activeTab === 'security' && (
          <section aria-labelledby="security-title">
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 id="security-title" className="card-title">
                    <Shield size={18} className="text-primary" />
                    Database-Level Security & Row-Level Security (RLS)
                  </h2>
                  <p className="card-desc">
                    Simulate real-time PostgreSQL authorization queries. Row-Level Security guarantees that vector similarity ranking respects patient privacy and departmental scoping inside the database kernel.
                  </p>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                <div
                  className={`card ${secRole === 'attending_physician' ? 'badge-primary' : ''}`}
                  style={{ cursor: 'pointer', padding: '1rem', border: secRole === 'attending_physician' ? '2px solid var(--primary)' : '1px solid var(--border)' }}
                  onClick={() => setSecRole('attending_physician')}
                >
                  <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-main)' }}>Attending Physician</h3>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Scoped strictly to department (e.g. Cardiology). Cross-department vector queries are blocked.
                  </p>
                </div>

                <div
                  className={`card ${secRole === 'radiologist' ? 'badge-primary' : ''}`}
                  style={{ cursor: 'pointer', padding: '1rem', border: secRole === 'radiologist' ? '2px solid var(--primary)' : '1px solid var(--border)' }}
                  onClick={() => setSecRole('radiologist')}
                >
                  <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-main)' }}>Diagnostic Radiologist</h3>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Permitted to view radiology reports and imaging assets. Non-imaging clinical documents blocked.
                  </p>
                </div>

                <div
                  className={`card ${secRole === 'clinical_researcher' ? 'badge-primary' : ''}`}
                  style={{ cursor: 'pointer', padding: '1rem', border: secRole === 'clinical_researcher' ? '2px solid var(--primary)' : '1px solid var(--border)' }}
                  onClick={() => setSecRole('clinical_researcher')}
                >
                  <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-main)' }}>Clinical Researcher</h3>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Access limited to de-identified view. Direct access to <code>patients</code> table rejected with InsufficientPrivilege.
                  </p>
                </div>

                <div
                  className={`card ${secRole === 'clinical_admin' ? 'badge-primary' : ''}`}
                  style={{ cursor: 'pointer', padding: '1rem', border: secRole === 'clinical_admin' ? '2px solid var(--primary)' : '1px solid var(--border)' }}
                  onClick={() => setSecRole('clinical_admin')}
                >
                  <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-main)' }}>Clinical Administrator</h3>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    Unrestricted full visibility across all relational tables and vector embeddings.
                  </p>
                </div>
              </div>

              {secRole === 'attending_physician' && (
                <div className="form-group" style={{ maxWidth: '280px' }}>
                  <label className="form-label">Active Physician Department Scope</label>
                  <select
                    className="form-select"
                    value={secDept}
                    onChange={(e) => setSecDept(e.target.value)}
                  >
                    <option value="Cardiology">Cardiology</option>
                    <option value="Pulmonology">Pulmonology</option>
                    <option value="Neurology">Neurology</option>
                    <option value="Oncology">Oncology</option>
                  </select>
                </div>
              )}

              <button
                type="button"
                className="btn btn-primary"
                onClick={handleSecurityTest}
                disabled={testingSec}
              >
                {testingSec ? <span className="spinner" /> : <Shield size={16} />}
                Execute Live Authorization Test ({secRole})
              </button>
            </div>

            {/* Security Audit Verification Cards */}
            {secReport && (
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">
                    PostgreSQL Kernel Enforcement Report: {secReport.role}
                  </h3>
                </div>

                <div style={{ display: 'grid', gap: '0.75rem' }}>
                  {secReport.checks?.map((chk, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '0.9rem',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--border)',
                        background: chk.status.includes('PROTECTED') || chk.status.includes('DENIED') || chk.status.includes('SCOPED') || chk.status === 'ALLOWED'
                          ? '#f8fafc'
                          : '#fef2f2'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.92rem' }}>{chk.name}</span>
                        <span
                          className={`badge ${
                            chk.status === 'ALLOWED' || chk.status === 'SCOPED'
                              ? 'badge-primary'
                              : chk.status.includes('PROTECTED') || chk.status.includes('DENIED')
                              ? 'badge-success'
                              : 'badge-danger'
                          }`}
                        >
                          {chk.status}
                        </span>
                      </div>
                      <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                        {chk.details}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* ================================================================= */}
        {/* TAB 4: CLINICAL RAG ASSISTANT                                     */}
        {/* ================================================================= */}
        {activeTab === 'rag' && (
          <section aria-labelledby="rag-title">
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 id="rag-title" className="card-title">
                    <Bot size={18} className="text-primary" />
                    Clinical RAG Demonstration & Decoupled Evaluation
                  </h2>
                  <p className="card-desc">
                    Retrieval-Augmented Generation prototype querying PostgreSQL pgvector. Synthesizes answers strictly from retrieved case summaries, requires explicit document citations, and triggers mandatory abstention when evidence is absent.
                  </p>
                </div>
              </div>

              <form onSubmit={handleRAGExecute}>
                <div className="form-group">
                  <label className="form-label">Research Inquiry</label>
                  <div className="search-input-group">
                    <input
                      type="text"
                      className="form-control search-input"
                      value={ragQuestion}
                      onChange={(e) => setRagQuestion(e.target.value)}
                      placeholder="e.g. What evidence exists regarding sarcoidosis and myocardial infarction?"
                    />
                    <button type="submit" className="btn btn-primary" disabled={runningRAG}>
                      {runningRAG ? <span className="spinner" /> : <Bot size={16} />}
                      Run RAG Pipeline
                    </button>
                  </div>

                  <div className="preset-queries">
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', alignSelf: 'center' }}>
                      Try Question:
                    </span>
                    {PRESET_RAG_QUESTIONS.map((pq, i) => (
                      <button
                        key={i}
                        type="button"
                        className="preset-chip"
                        onClick={() => setRagQuestion(pq.text)}
                      >
                        {pq.label}
                      </button>
                    ))}
                  </div>
                </div>
              </form>
            </div>

            {/* RAG Answer & Evidence Display */}
            {ragResult && (
              <div>
                <div className="card" style={{ borderLeft: '4px solid var(--primary)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                    <span className="badge badge-primary">
                      Evidence Status: {ragResult.evidence_status}
                    </span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Total Latency: {ragResult.timing_ms?.total} ms (Retrieval: {ragResult.timing_ms?.retrieval} ms)
                    </span>
                  </div>

                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.92rem' }}>
                    {ragResult.answer}
                  </div>

                  {ragResult.citations?.length > 0 && (
                    <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border)' }}>
                      <span style={{ fontSize: '0.82rem', fontWeight: 700, marginRight: '0.5rem' }}>
                        Cited Document Identifiers:
                      </span>
                      {ragResult.citations.map((c, i) => (
                        <span key={i} className="badge badge-indigo" style={{ marginRight: '4px' }}>
                          [{c}]
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Decoupled Academic Evaluation Report */}
                {ragResult.evaluation && (
                  <div className="card" style={{ background: '#f8fafc' }}>
                    <h3 className="card-title" style={{ fontSize: '0.95rem', marginBottom: '0.75rem' }}>
                      Decoupled Academic Evaluation Report
                    </h3>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                      <div style={{ padding: '0.85rem', background: 'white', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
                        <h4 style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '0.5rem' }}>
                          [Part 1: Retrieval Quality]
                        </h4>
                        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                          <div>Returned Passages: <strong>{ragResult.evaluation.retrieval_quality?.returned_count}</strong></div>
                          <div>Best Cosine Distance: <strong>{ragResult.evaluation.retrieval_quality?.top_distance}</strong></div>
                          <div>Query Lexical Overlap: <strong>{(ragResult.evaluation.retrieval_quality?.lexical_overlap_ratio * 100).toFixed(1)}%</strong></div>
                          <div>Adequate Evidence: <strong>{String(ragResult.evaluation.retrieval_quality?.has_adequate_evidence)}</strong></div>
                        </div>
                      </div>

                      <div style={{ padding: '0.85rem', background: 'white', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
                        <h4 style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '0.5rem' }}>
                          [Part 2: Generation Groundedness]
                        </h4>
                        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                          <div>Citation Precision: <strong>{(ragResult.evaluation.generation_groundedness?.citation_precision * 100).toFixed(0)}%</strong></div>
                          <div>Refusal Protocol Status: <span className="badge badge-success">{ragResult.evaluation.generation_groundedness?.refusal_correctness}</span></div>
                          <div>Unsupported Claims: <strong>{ragResult.evaluation.generation_groundedness?.unsupported_claims?.length || 0}</strong></div>
                          <div>Overall Groundedness Score: <strong>{ragResult.evaluation.generation_groundedness?.overall_groundedness_score} / 1.00</strong></div>
                        </div>
                      </div>
                    </div>

                    <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '0.75rem', fontStyle: 'italic' }}>
                      {ragResult.disclaimer}
                    </p>
                  </div>
                )}
              </div>
            )}
          </section>
        )}

        {/* ================================================================= */}
        {/* TAB 5: DATABASE ARCHITECTURE & TELEMETRY                          */}
        {/* ================================================================= */}
        {activeTab === 'telemetry' && (
          <section aria-labelledby="telemetry-title">
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 id="telemetry-title" className="card-title">
                    <Activity size={18} className="text-primary" />
                    Database Schema & Storage Architecture Audit
                  </h2>
                  <p className="card-desc">
                    Live relational table counts, physical relation sizing, and pgvector index configurations inside PostgreSQL 18.6.
                  </p>
                </div>
                <button type="button" className="btn btn-secondary btn-sm" onClick={fetchStats} disabled={loadingStats}>
                  {loadingStats ? <span className="spinner" /> : 'Refresh Catalog'}
                </button>
              </div>

              {/* Accessible Responsive Table conforming to modern-web-guidance */}
              <div
                tabIndex={0}
                role="region"
                aria-label="PostgreSQL Table Storage Audit"
                className="table-wrapper"
              >
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Table Relation</th>
                      <th scope="col">Stored Rows</th>
                      <th scope="col">Relation Size</th>
                      <th scope="col">Schema Normalization & Role</th>
                    </tr>
                  </thead>
                  <tbody>
                    {systemStats?.tables?.map((t) => (
                      <tr key={t.table}>
                        <th scope="row">{t.table}</th>
                        <td><strong>{t.count.toLocaleString()}</strong></td>
                        <td><span className="metric-pill gray">{t.size}</span></td>
                        <td style={{ color: 'var(--text-muted)' }}>
                          {t.table === 'patients' && '3NF Primary Demographics (MRN, DOB, Gender)'}
                          {t.table === 'encounters' && 'Clinical Visits (Department, Encounter Type, Date)'}
                          {t.table === 'diagnoses' && 'ICD-10 Diagnostic Codes & Billing Data'}
                          {t.table === 'clinical_documents' && 'Physician Case Reports & PubMed PMIDs'}
                          {t.table === 'document_chunks' && 'Sliding-Window Excerpts with SHA-256 Provenance'}
                          {t.table === 'chunk_embeddings' && 'vector(384) High-Dimensional Cosine Embeddings'}
                          {t.table === 'medical_images' && 'Radiology Assets (CT, DX, US) with SHA-256 Hashes'}
                          {t.table === 'image_embeddings' && 'vector(512) Dense Multimodal Vision Embeddings'}
                          {t.table === 'embedding_models' && 'Model Registry (Sentence-Transformers & CLIP)'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Active Indexes */}
              <div style={{ marginTop: '1.25rem' }}>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                  Vector Indexing Relations (pgvector)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem' }}>
                  <div style={{ padding: '0.85rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.88rem' }}>idx_chunk_embeddings_hnsw_cosine</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                      HNSW Proximity Graph (m=16, ef_c=64) • Size: <strong>12 MB</strong>
                    </div>
                  </div>
                  <div style={{ padding: '0.85rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontWeight: 700, fontSize: '0.88rem' }}>idx_chunk_embeddings_ivfflat_cosine</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                      IVFFlat Inverted File (lists=25) • Size: <strong>10,128 kB</strong>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval • PostgreSQL 18.6 • pgvector 0.8.6 • PubMed Central (PMC-Patients, CC-BY 4.0)
        </p>
      </footer>
    </div>
  );
}
