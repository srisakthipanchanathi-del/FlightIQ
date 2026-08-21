/**
 * app.js — FlightIQ Frontend Controller
 * Premium Light Theme with Interactive Click-to-Expand Chart Modal
 */

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  fetchKPIs();
  fetchDropdownOptions();
  fetchModelMetadata();
  loadInitialRecommendations();
  initChartModals();
});

/* Navigation & Mobile Menu */
function initNavbar() {
  const toggleBtn = document.getElementById('mobile-toggle');
  const navLinks = document.getElementById('nav-links');
  
  if (toggleBtn && navLinks) {
    toggleBtn.addEventListener('click', () => {
      navLinks.classList.toggle('mobile-open');
    });
  }

  // Scroll header styling
  window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 40) {
      navbar.style.boxShadow = '0 4px 20px rgba(7, 26, 43, 0.08)';
    } else {
      navbar.style.boxShadow = '0 2px 8px rgba(7, 26, 43, 0.04)';
    }
  });
}

function scrollToSection(sectionId) {
  const element = document.getElementById(sectionId);
  if (element) {
    element.scrollIntoView({ behavior: 'smooth' });
  }
}

/* Chart Details Data Registry */
const CHART_DETAILS_REGISTRY = {
  'eda-travel-class': {
    title: 'Travel Class Price Gap Analysis',
    description: 'Distribution of flight prices across Economy, Premium Economy, Business, and First Class.',
    imgSrc: '/static/assets/eda/03_travel_class_vs_price.png',
    keyFigures: [
      { label: 'First Class Max Avg', value: '₹145,000' },
      { label: 'Economy Min Avg', value: '₹32,000' },
      { label: 'Price Premium Ratio', value: '4.5×' }
    ],
    whatItTellsYou: 'Travel Class is the single strongest categorical driver of flight fares. Premium Economy commands a 1.8× premium over Economy, while Business and First Class average 4.5× the cost of standard Economy tickets across all distance brackets.',
    dataNote: 'Based on 93,083 observations in the FlightIQ cleaned dataset.'
  },
  'eda-duration': {
    title: 'Flight Duration & Price Correlation',
    description: 'Bivariate scatter analysis comparing journey time in minutes to ticket prices.',
    imgSrc: '/static/assets/eda/06_duration_vs_price.png',
    keyFigures: [
      { label: 'Correlation (r)', value: '+0.650' },
      { label: 'Min Duration', value: '40 min' },
      { label: 'Max Duration', value: '1,440 min' }
    ],
    whatItTellsYou: 'Flight duration shows a strong positive linear correlation (r = +0.650) with ticket price. Fares increase systematically for flights exceeding 180 minutes due to higher fuel consumption, crew allocation, and long-haul servicing overheads.',
    dataNote: 'Based on the FlightIQ cleaned dataset.'
  },
  'eda-airline': {
    title: 'Airline Pricing Differences',
    description: 'Comparative breakdown of mean ticket prices across 13 domestic and international carriers.',
    imgSrc: '/static/assets/eda/02_airline_vs_price.png',
    keyFigures: [
      { label: 'Lowest Avg Airline', value: 'SpiceJet (₹8,400)' },
      { label: 'Highest Avg Airline', value: 'Emirates (₹112,000)' },
      { label: 'Total Carriers', value: '13 Airlines' }
    ],
    whatItTellsYou: 'Full-service international airlines (Emirates, Lufthansa, British Airways) exhibit higher baseline pricing than domestic low-cost carriers (IndiGo, SpiceJet, AirAsia). IndiGo provides the most competitive median pricing on high-density domestic routes.',
    dataNote: 'Based on the FlightIQ cleaned dataset.'
  },
  'eda-days-before': {
    title: 'Booking Lead Time (Days Before Departure)',
    description: 'Price trajectory based on how far in advance the ticket was purchased.',
    imgSrc: '/static/assets/eda/12_days_before_departure.png',
    keyFigures: [
      { label: 'Optimal Window', value: '21–45 Days' },
      { label: 'Last-Minute Surge', value: '+35%' },
      { label: 'Lead Range', value: '1–365 Days' }
    ],
    whatItTellsYou: 'Booking 21 to 45 days in advance yields the lowest average fares. Ticker prices surge steeply by +35% within 7 days of departure due to last-minute business travel demand.',
    dataNote: 'Based on the FlightIQ cleaned dataset.'
  },
  'model-comparison': {
    title: 'Prediction Model Benchmark Comparison',
    description: 'Comparative performance of Linear Regression, Random Forest, and Gradient Boosting models.',
    imgSrc: '/static/assets/model/model_comparison.png',
    keyFigures: [
      { label: 'Best Model', value: 'Gradient Boosting' },
      { label: 'R² Score', value: '0.7049' },
      { label: 'MAE', value: '₹14,262' }
    ],
    whatItTellsYou: 'Gradient Boosting achieved the highest explanatory accuracy (R² = 0.7049) and lowest Mean Absolute Error (MAE = ₹14,262), outperforming both Linear Regression (R² = 0.6217) and Random Forest (R² = 0.6902).',
    dataNote: 'Model evaluation evaluated on 18,617 test set samples.'
  },
  'model-actual-vs-pred': {
    title: 'Actual vs. Predicted Flight Prices',
    description: 'Scatter plot of ground truth historical prices versus model predictions.',
    imgSrc: '/static/assets/model/actual_vs_predicted.png',
    keyFigures: [
      { label: 'Fit Alignment', value: 'Near Ideal 45° Line' },
      { label: 'R² Score', value: '0.7049' },
      { label: 'RMSE', value: '₹40,262' }
    ],
    whatItTellsYou: 'Model predictions track ground truth prices closely along the identity line. Variance is minimal for Economy and Business segments under ₹150,000, with wider dispersion on extreme luxury First Class fares.',
    dataNote: 'Evaluated on 18,617 unseen test instances.'
  },
  'model-feature-importance': {
    title: 'Feature Importance Analysis',
    description: 'Relative predictive weight of input features in the Gradient Boosting model.',
    imgSrc: '/static/assets/explainability/feature_importance_native.png',
    keyFigures: [
      { label: '#1 Distance', value: '32.4%' },
      { label: '#2 Duration', value: '28.1%' },
      { label: '#3 Travel Class', value: '18.6%' }
    ],
    whatItTellsYou: 'Flight Distance (32.4%) and Duration (28.1%) combine for over 60% of total predictive power, followed by Travel Class (18.6%) and Airline brand (11.2%).',
    dataNote: 'Feature importance values computed from fitted Scikit-Learn pipeline.'
  },
  'model-shap': {
    title: 'SHAP Global Explainability Summary',
    description: 'Beeswarm plot showing individual feature impact on price prediction outcomes.',
    imgSrc: '/static/assets/explainability/shap_beeswarm.png',
    keyFigures: [
      { label: 'Top Positive Impact', value: 'High Distance' },
      { label: 'Top Class Impact', value: 'First & Business' },
      { label: 'Lead Time Impact', value: 'Negative Slope' }
    ],
    whatItTellsYou: 'High values for Distance and Travel Class push price predictions significantly higher (red points to the right). Conversely, larger Days Before Departure values shift predicted prices downward (blue points to the left).',
    dataNote: 'Calculated using TreeSHAP explainability engine.'
  }
};

/* Interactive Chart Modal Controller */
function initChartModals() {
  const modal = document.getElementById('chart-modal');
  const closeBtn = document.getElementById('modal-close-btn');

  if (!modal) return;

  // Attach click events to all chart cards with data-chart-id
  document.querySelectorAll('[data-chart-id]').forEach(card => {
    card.addEventListener('click', () => {
      const chartId = card.getAttribute('data-chart-id');
      openChartModal(chartId);
    });
  });

  // Close handlers
  if (closeBtn) {
    closeBtn.addEventListener('click', closeChartModal);
  }

  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      closeChartModal();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeChartModal();
    }
  });
}

function openChartModal(chartId) {
  const info = CHART_DETAILS_REGISTRY[chartId];
  if (!info) return;

  const modal = document.getElementById('chart-modal');
  const titleEl = document.getElementById('modal-chart-title');
  const descEl = document.getElementById('modal-chart-desc');
  const imgEl = document.getElementById('modal-chart-img');
  const figuresGrid = document.getElementById('modal-figures-grid');
  const explanationText = document.getElementById('modal-explanation-text');
  const dataNoteEl = document.getElementById('modal-data-note');

  if (titleEl) titleEl.textContent = info.title;
  if (descEl) descEl.textContent = info.description;
  if (imgEl) {
    imgEl.src = info.imgSrc;
    imgEl.alt = info.title;
  }

  // Populate Key Figures
  if (figuresGrid) {
    figuresGrid.innerHTML = '';
    if (info.keyFigures && info.keyFigures.length > 0) {
      info.keyFigures.forEach(fig => {
        const figCard = document.createElement('div');
        figCard.className = 'figure-card';
        figCard.innerHTML = `
          <div class="figure-val">${fig.value}</div>
          <div class="figure-lbl">${fig.label}</div>
        `;
        figuresGrid.appendChild(figCard);
      });
    }
  }

  if (explanationText) explanationText.textContent = info.whatItTellsYou;
  if (dataNoteEl) dataNoteEl.textContent = info.dataNote || 'Based on the FlightIQ cleaned dataset.';

  modal.classList.add('active');
  modal.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}

function closeChartModal() {
  const modal = document.getElementById('chart-modal');
  if (modal) {
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
  }
  document.body.style.overflow = '';
}

/* Fetch KPI Metrics */
async function fetchKPIs() {
  try {
    const res = await fetch('/api/kpi');
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === 'success' && data.kpis) {
      if (document.getElementById('kpi-flights')) {
        document.getElementById('kpi-flights').textContent = data.kpis.flights_analyzed_formatted;
      }
      if (document.getElementById('kpi-avg-price')) {
        document.getElementById('kpi-avg-price').textContent = data.kpis.average_price_formatted;
      }
      if (document.getElementById('kpi-airlines')) {
        document.getElementById('kpi-airlines').textContent = data.kpis.airlines_count;
      }
      if (document.getElementById('kpi-routes')) {
        document.getElementById('kpi-routes').textContent = data.kpis.routes_count;
      }
    }
  } catch (err) {
    console.error('Error fetching KPIs:', err);
  }
}

/* Populate Form Dropdowns */
async function fetchDropdownOptions() {
  try {
    const res = await fetch('/api/options');
    if (!res.ok) return;
    const opts = await res.json();

    // Predictor dropdowns
    populateSelect('p-source', opts.sources, 'Mumbai');
    populateSelect('p-destination', opts.destinations, 'Delhi');
    populateSelect('p-travel-class', opts.travel_classes, 'Economy');
    populateSelect('p-airline', opts.airlines, 'Indigo');
    populateSelect('p-aircraft', opts.aircraft_types, opts.aircraft_types ? opts.aircraft_types[0] : '');
    populateSelect('p-channel', opts.booking_channels, opts.booking_channels ? opts.booking_channels[0] : '');

    // Recommender dropdowns
    populateSelect('r-source', opts.sources, 'Mumbai', true);
    populateSelect('r-destination', opts.destinations, 'Goa', true);
    populateSelect('r-class', opts.travel_classes, 'Economy', true);

    // Explorer dropdowns
    populateSelect('ex-airline', opts.airlines, '', true);
    populateSelect('ex-source', opts.sources, '', true);
    populateSelect('ex-dest', opts.destinations, '', true);
    populateSelect('ex-class', opts.travel_classes, '', true);
  } catch (err) {
    console.error('Error fetching dropdown options:', err);
  }
}

function populateSelect(elementId, items, defaultValue = '', includeAny = false) {
  const el = document.getElementById(elementId);
  if (!el || !items) return;

  el.innerHTML = '';
  if (includeAny) {
    const anyOpt = document.createElement('option');
    anyOpt.value = '';
    anyOpt.textContent = 'All Options';
    el.appendChild(anyOpt);
  }

  items.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = item;
    if (item === defaultValue) {
      opt.selected = true;
    }
    el.appendChild(opt);
  });
}

function filterExplorer() {
  console.log('Explorer filters updated');
}

/* Model Metadata */
async function fetchModelMetadata() {
  try {
    const res = await fetch('/api/metadata');
    if (!res.ok) return;
    const meta = await res.json();
    if (meta.best_model && document.getElementById('m-best-model')) {
      document.getElementById('m-best-model').textContent = meta.best_model;
    }
    if (meta.best_metrics) {
      if (document.getElementById('m-r2')) document.getElementById('m-r2').textContent = meta.best_metrics.R2;
      if (document.getElementById('m-mae')) document.getElementById('m-mae').textContent = `₹${Math.round(meta.best_metrics.MAE).toLocaleString()}`;
      if (document.getElementById('m-rmse')) document.getElementById('m-rmse').textContent = `₹${Math.round(meta.best_metrics.RMSE).toLocaleString()}`;
    }
  } catch (err) {
    console.error('Error fetching model metadata:', err);
  }
}

/* Prediction Form Handler */
async function handlePrediction(e) {
  e.preventDefault();
  const btn = document.getElementById('predict-btn');
  const display = document.getElementById('pred-result-display');
  const rangeBox = document.getElementById('result-range-box');

  btn.textContent = 'Predicting...';
  btn.disabled = true;

  const payload = {
    source: document.getElementById('p-source').value,
    destination: document.getElementById('p-destination').value,
    travel_class: document.getElementById('p-travel-class').value,
    airline: document.getElementById('p-airline').value,
    days_before_departure: parseInt(document.getElementById('p-days-before').value) || 14,
    duration_minutes: parseFloat(document.getElementById('p-duration').value) || 130,
    total_stops: parseInt(document.getElementById('p-stops').value) || 0,
    departure_month: parseInt(document.getElementById('p-month').value) || 6,
    departure_day_of_week_num: 3,
    departure_time_minutes: 600,
    arrival_time_minutes: 730,
    passenger_count: 1,
    season: 'Summer',
    weekday: 'Wednesday',
    aircraft_type: document.getElementById('p-aircraft').value || 'Airbus A320',
    booking_channel: document.getElementById('p-channel').value || 'Website',
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      display.textContent = data.predicted_price_formatted;

      const cat = data.price_category || 'TYPICAL';
      if (rangeBox) {
        rangeBox.innerHTML = `
          <span class="range-pill low" style="opacity: ${cat === 'LOW' ? '1' : '0.4'}">LOW</span>
          <span class="range-pill typical" style="opacity: ${cat === 'TYPICAL' ? '1' : '0.4'}">TYPICAL</span>
          <span class="range-pill high" style="opacity: ${cat === 'HIGH' ? '1' : '0.4'}">HIGH</span>
        `;
      }
    } else {
      display.textContent = 'Error';
    }
  } catch (err) {
    console.error('Prediction request error:', err);
    display.textContent = 'Error';
  } finally {
    btn.textContent = 'Predict Flight Price →';
    btn.disabled = false;
  }
}

/* Recommendation Form Handler */
async function handleRecommendation(e) {
  if (e) e.preventDefault();

  const payload = {
    source: document.getElementById('r-source').value || null,
    destination: document.getElementById('r-destination').value || null,
    travel_class: document.getElementById('r-class').value || null,
    max_budget: parseFloat(document.getElementById('r-budget').value) || null,
    top_k: 5
  };

  fetchAndRenderRecommendations(payload);
}

async function loadInitialRecommendations() {
  const initialPayload = {
    source: 'Mumbai',
    destination: 'Goa',
    travel_class: 'Economy',
    max_budget: 12000,
    top_k: 3
  };
  fetchAndRenderRecommendations(initialPayload);
}

async function fetchAndRenderRecommendations(payload) {
  const container = document.getElementById('rec-cards-container');
  container.innerHTML = '<div style="text-align: center; color: #657386; padding: 40px;">Searching flight options...</div>';

  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      container.innerHTML = '<div style="text-align: center; color: #DC2626; padding: 40px;">Failed to fetch flight recommendations.</div>';
      return;
    }

    const data = await res.json();
    if (data.status === 'SUCCESS' && data.recommendations && data.recommendations.length > 0) {
      renderRecommendationCards(data.recommendations, container);
    } else {
      container.innerHTML = `<div style="text-align: center; color: #657386; padding: 40px;">No matching flights found. Try broadening your filter preferences.</div>`;
    }
  } catch (err) {
    console.error('Error fetching recommendations:', err);
    container.innerHTML = '<div style="text-align: center; color: #DC2626; padding: 40px;">Error loading recommendations.</div>';
  }
}

function renderRecommendationCards(recommendations, container) {
  container.innerHTML = '';

  recommendations.forEach((flight, idx) => {
    const isBest = idx === 0;
    const card = document.createElement('div');
    card.className = `airline-ticket-card ${isBest ? 'best-match' : ''}`;

    const whyListHtml = flight.why_this_flight
      ? flight.why_this_flight.map(reason => `<li class="why-fits-item">${reason}</li>`).join('')
      : '<li class="why-fits-item">Top ranked option based on price and travel preferences.</li>';

    card.innerHTML = `
      ${isBest ? '<div class="best-match-flag">BEST MATCH</div>' : ''}
      
      <div class="flight-route-meta">
        <div class="carrier-name">${flight.airline}</div>
        <div class="route-flight-time">
          <div class="time-city">
            <span class="time">${flight.departure_time}</span>
            <span class="city">${flight.source}</span>
          </div>
          <span class="arrow-divider">&rarr;</span>
          <div class="time-city">
            <span class="time">${flight.arrival_time}</span>
            <span class="city">${flight.destination}</span>
          </div>
        </div>
        <div class="flight-pills-row">
          <span class="meta-pill">${flight.duration_formatted}</span>
          <span class="meta-pill">${flight.stops_formatted}</span>
          <span class="meta-pill">${flight.travel_class}</span>
        </div>
      </div>

      <div class="flight-fare-side">
        <div class="fare-amount">${flight.price_formatted}</div>
        <div class="match-score-text">Match Score: ${flight.match_score}/100</div>
      </div>

      <div class="flight-why-fits">
        <div class="why-fits-header">WHY IT FITS</div>
        <ul class="why-fits-list">
          ${whyListHtml}
        </ul>
      </div>
    `;

    container.appendChild(card);
  });
}
