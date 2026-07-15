import { useState } from 'react';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    GA_By_DatesWeeks: '',
    BirthWeight: '',
    Multiple_gest: 'no',
    Sex: 'female',
    Apgar_5: '',
    PC_Preterm_Labor: 'no',
    ivh_severe: 'no',
    Race: 'Others'
  });

  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setScore(null);

    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          GA_By_DatesWeeks: parseFloat(formData.GA_By_DatesWeeks),
          BirthWeight: parseFloat(formData.BirthWeight),
          Apgar_5: parseFloat(formData.Apgar_5)
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      const data = await response.json();
      setScore(data.Score);
    } catch (err) {
      setError('Failed to connect to the prediction API. Is the server running on port 5001?');
    } finally {
      setLoading(false);
    }
  };

  const percentage = score !== null ? (score * 100).toFixed(1) : 0;
  
  // Determine color based on risk (low risk = green, medium = yellow, high = red)
  let gaugeColor = 'var(--success)';
  if (score > 0.3) gaugeColor = '#eab308'; // yellow
  if (score > 0.6) gaugeColor = 'var(--danger)'; // red

  return (
    <div className="glass-panel">
      <h2>SIP Risk Prediction</h2>
      <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
        Enter patient metrics to calculate the probability of Spontaneous Intestinal Perforation.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="input-group">
            <label htmlFor="GA_By_DatesWeeks">Gestational Age (Weeks)</label>
            <input 
              type="number" 
              step="0.1"
              id="GA_By_DatesWeeks" 
              name="GA_By_DatesWeeks" 
              value={formData.GA_By_DatesWeeks} 
              onChange={handleChange} 
              required 
              placeholder="e.g. 28" 
            />
          </div>

          <div className="input-group">
            <label htmlFor="BirthWeight">Birth Weight (grams)</label>
            <input 
              type="number" 
              id="BirthWeight" 
              name="BirthWeight" 
              value={formData.BirthWeight} 
              onChange={handleChange} 
              required 
              placeholder="e.g. 1240" 
            />
          </div>

          <div className="input-group">
            <label htmlFor="Apgar_5">Apgar Score (5 mins)</label>
            <input 
              type="number" 
              id="Apgar_5" 
              name="Apgar_5" 
              value={formData.Apgar_5} 
              onChange={handleChange} 
              required 
              placeholder="e.g. 8" 
              min="0"
              max="10"
            />
          </div>

          <div className="input-group">
            <label htmlFor="Sex">Sex</label>
            <select id="Sex" name="Sex" value={formData.Sex} onChange={handleChange}>
              <option value="female">Female</option>
              <option value="male">Male</option>
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="Multiple_gest">Multiple Gestation</label>
            <select id="Multiple_gest" name="Multiple_gest" value={formData.Multiple_gest} onChange={handleChange}>
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="PC_Preterm_Labor">Preterm Labor</label>
            <select id="PC_Preterm_Labor" name="PC_Preterm_Labor" value={formData.PC_Preterm_Labor} onChange={handleChange}>
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="ivh_severe">Severe IVH</label>
            <select id="ivh_severe" name="ivh_severe" value={formData.ivh_severe} onChange={handleChange}>
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </div>

          <div className="input-group">
            <label htmlFor="Race">Race</label>
            <select id="Race" name="Race" value={formData.Race} onChange={handleChange}>
              <option value="W">White</option>
              <option value="B">Black</option>
              <option value="A">Asian</option>
              <option value="O">Other</option>
              <option value="U">Unknown</option>
              <option value="Others">Others (Grouped)</option>
            </select>
          </div>
        </div>

        {error && (
          <div style={{ color: 'var(--danger)', marginBottom: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px' }}>
            {error}
          </div>
        )}

        <button type="submit" className="btn-submit" disabled={loading}>
          {loading ? <span className="spinner"></span> : 'Calculate Risk Score'}
        </button>
      </form>

      {score !== null && !error && (
        <div className="result-card">
          <div className="result-title">Predicted SIP Probability</div>
          <div className="result-score">{percentage}%</div>
          <div className="result-gauge">
            <div 
              className="gauge-fill" 
              style={{ 
                width: `${percentage}%`,
                backgroundColor: gaugeColor
              }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
