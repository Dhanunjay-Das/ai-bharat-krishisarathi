# Requirements Document: KrishiSarathi (Farmer's Guide)

## Introduction

KrishiSarathi is an AI-powered agricultural assistant designed to bridge the language, digital, and information gaps faced by Indian farmers. The system provides voice-first, multi-channel access to critical agricultural services including disease diagnosis, government scheme navigation, market intelligence, and weather alerts in 8 Indian languages. The system must work across feature phones (IVR) and smartphones (Web/WhatsApp) to reach farmers with varying levels of digital access.

## Glossary

- **KrishiSarathi**: The AI-powered agricultural assistant system
- **Voice_Interface**: The speech recognition and synthesis component that handles voice input/output
- **Disease_Diagnoser**: The AI component that analyzes crop symptoms and provides disease identification
- **Scheme_Navigator**: The component that determines government scheme eligibility
- **Market_Intelligence**: The component that provides real-time agricultural commodity prices
- **Weather_Service**: The component that delivers location-based weather forecasts and farming advice
- **User**: A farmer or agricultural stakeholder using the system
- **Regional_Language**: One of the 8 supported Indian languages (Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, Gujarati, Punjabi)
- **Mandi**: Agricultural market/trading center
- **IVR**: Interactive Voice Response system for feature phones
- **Confidence_Score**: A numerical value (0-100%) indicating AI model certainty
- **Expert**: A human agricultural specialist for complex case escalation
- **Farming_Hours**: The operational window from 5 AM to 8 PM IST

## Requirements

### Requirement 1: Voice-Based Multi-Language Interface

**User Story:** As Ram Singh (wheat farmer, Hindi speaker with basic phone), I want to speak my agricultural questions in Hindi and receive voice responses, so that I can access agricultural information without needing to read English or use complex interfaces.

#### Acceptance Criteria

1. WHEN a User speaks in any Regional_Language, THE Voice_Interface SHALL recognize the speech and convert it to text with at least 85% accuracy
2. WHEN the Voice_Interface processes text responses, THE Voice_Interface SHALL synthesize natural-sounding speech in the User's chosen Regional_Language
3. WHEN a User speaks with regional accents or colloquial terms, THE Voice_Interface SHALL use AI-based speech recognition to understand context-specific agricultural terminology
4. THE Voice_Interface SHALL support all 8 Regional_Languages (Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, Gujarati, Punjabi)
5. WHEN speech recognition confidence is below 70%, THE Voice_Interface SHALL ask the User to repeat or clarify their input

### Requirement 2: AI-Powered Disease Diagnosis

**User Story:** As Ram Singh (wheat farmer experiencing crop issues), I want to describe my crop symptoms in my own words and receive disease diagnosis with treatment recommendations, so that I can take timely action to prevent crop losses.

#### Acceptance Criteria

1. WHEN a User describes crop symptoms via voice or text, THE Disease_Diagnoser SHALL analyze the symptoms using AI models and return a ranked list of possible diseases with Confidence_Scores
2. WHEN multiple diseases match the symptoms, THE Disease_Diagnoser SHALL consider contextual factors (location, season, soil type, crop variety) to refine the diagnosis
3. WHEN providing a diagnosis, THE Disease_Diagnoser SHALL include the Confidence_Score, treatment recommendations, and preventive measures
4. WHEN the highest Confidence_Score is below 60%, THE Disease_Diagnoser SHALL escalate the case to an Expert and notify the User of the escalation
5. WHEN providing treatment recommendations, THE Disease_Diagnoser SHALL explain in simple language why this diagnosis was suggested based on the symptoms provided
6. THE Disease_Diagnoser SHALL support diagnosis for at least 50 common crop diseases across major Indian crops (wheat, rice, cotton, sugarcane, pulses)

### Requirement 3: Government Scheme Eligibility Navigator

**User Story:** As Lakshmi (rice farmer seeking financial assistance), I want to answer simple questions about my farm and situation to discover which government schemes I'm eligible for, so that I can access available benefits without navigating complex bureaucracy.

#### Acceptance Criteria

1. WHEN a User requests scheme information, THE Scheme_Navigator SHALL ask qualifying questions in the User's Regional_Language to determine eligibility
2. WHEN the User answers all qualifying questions, THE Scheme_Navigator SHALL return a list of eligible government schemes with application procedures
3. THE Scheme_Navigator SHALL cover at least 20 major central and state agricultural schemes (PM-KISAN, Crop Insurance, Kisan Credit Card, etc.)
4. WHEN displaying scheme information, THE Scheme_Navigator SHALL include scheme name, benefits, eligibility criteria, required documents, and application links
5. WHEN a User is not eligible for any schemes, THE Scheme_Navigator SHALL suggest alternative resources or upcoming schemes they might qualify for

### Requirement 4: Real-Time Market Intelligence

**User Story:** As Lakshmi (rice farmer planning to sell harvest), I want to know current market prices at nearby mandis, so that I can decide the best time and place to sell my produce for maximum profit.

#### Acceptance Criteria

1. WHEN a User requests market prices for a crop, THE Market_Intelligence SHALL return real-time prices from the 5 nearest Mandis based on User location
2. WHEN displaying prices, THE Market_Intelligence SHALL show the crop name, price per quintal, Mandi name, distance from User, and last updated timestamp
3. THE Market_Intelligence SHALL update price data at least every 4 hours during Farming_Hours
4. WHEN price data is older than 24 hours, THE Market_Intelligence SHALL display a staleness warning to the User
5. THE Market_Intelligence SHALL support price queries for at least 30 major agricultural commodities

### Requirement 5: Location-Based Weather Alerts and Farming Advice

**User Story:** As Ram Singh (wheat farmer planning field operations), I want to receive weather forecasts and farming advice specific to my location and crop, so that I can plan irrigation, spraying, and harvesting at optimal times.

#### Acceptance Criteria

1. WHEN a User requests weather information, THE Weather_Service SHALL provide a 7-day forecast for the User's location including temperature, rainfall probability, humidity, and wind speed
2. WHEN weather conditions pose risks to crops (heavy rain, frost, heatwave), THE Weather_Service SHALL proactively send alerts to affected Users
3. WHEN providing weather forecasts, THE Weather_Service SHALL include crop-specific farming advice (e.g., "Good day for spraying pesticides - low wind and no rain expected")
4. THE Weather_Service SHALL update forecasts at least twice daily during Farming_Hours
5. WHEN extreme weather events are predicted (storms, floods, droughts), THE Weather_Service SHALL send urgent alerts at least 24 hours in advance

### Requirement 6: Multi-Channel Access

**User Story:** As a User with varying levels of digital access, I want to access KrishiSarathi through my available device (feature phone, smartphone, or computer), so that I can get agricultural assistance regardless of my technology constraints.

#### Acceptance Criteria

1. THE KrishiSarathi SHALL provide a web interface accessible via standard browsers on smartphones and computers
2. WHERE a User has a feature phone, THE KrishiSarathi SHALL provide IVR-based access via phone calls
3. WHERE a User has WhatsApp, THE KrishiSarathi SHALL provide a WhatsApp bot interface
4. WHEN a User switches channels (e.g., from web to IVR), THE KrishiSarathi SHALL maintain conversation context for at least 24 hours
5. THE KrishiSarathi SHALL provide consistent functionality across all channels with channel-appropriate interfaces

### Requirement 7: Performance and Responsiveness

**User Story:** As a User with limited patience and potentially slow internet, I want quick responses to my queries, so that I can get timely information without frustration or excessive wait times.

#### Acceptance Criteria

1. WHEN a User submits a query, THE KrishiSarathi SHALL return a response within 5 seconds for 95% of requests
2. WHEN processing takes longer than 5 seconds, THE KrishiSarathi SHALL display a progress indicator and estimated wait time
3. WHEN network connectivity is poor, THE KrishiSarathi SHALL optimize data transfer to work on 2G connections with at least basic functionality
4. THE KrishiSarathi SHALL support at least 10,000 concurrent Users without performance degradation
5. WHEN system load exceeds capacity, THE KrishiSarathi SHALL queue requests and inform Users of expected wait times rather than failing

### Requirement 8: High Availability During Critical Hours

**User Story:** As a User who works in the fields during specific hours, I want the system to be reliably available during farming hours, so that I can access information when I need it most.

#### Acceptance Criteria

1. THE KrishiSarathi SHALL maintain 99% uptime during Farming_Hours (5 AM to 8 PM IST)
2. WHEN system maintenance is required, THE KrishiSarathi SHALL schedule it outside Farming_Hours
3. WHEN unplanned outages occur during Farming_Hours, THE KrishiSarathi SHALL restore service within 30 minutes
4. THE KrishiSarathi SHALL provide a status page showing current system health and any known issues
5. WHEN the system is unavailable, THE KrishiSarathi SHALL display an informative message with expected restoration time and alternative contact methods

### Requirement 9: Transparent and Explainable AI

**User Story:** As Ram Singh (farmer receiving AI recommendations), I want to understand why the system gave me specific advice, so that I can trust the recommendations and make informed decisions about my crops.

#### Acceptance Criteria

1. WHEN the Disease_Diagnoser provides a diagnosis, THE KrishiSarathi SHALL explain which symptoms led to this conclusion in simple language
2. WHEN the AI model makes a recommendation, THE KrishiSarathi SHALL display the Confidence_Score prominently
3. WHEN a User requests more details, THE KrishiSarathi SHALL provide information about what data the AI considered (location, season, historical patterns)
4. THE KrishiSarathi SHALL never present AI recommendations as absolute certainty, always acknowledging the probabilistic nature
5. WHEN AI recommendations conflict with traditional farming practices, THE KrishiSarathi SHALL acknowledge both perspectives and explain the reasoning

### Requirement 10: Bias Mitigation and Fairness

**User Story:** As a Government Officer overseeing agricultural AI systems, I want to ensure the system serves all farmers fairly regardless of region, crop type, or language, so that no farming community is disadvantaged.

#### Acceptance Criteria

1. THE KrishiSarathi SHALL undergo quarterly audits to detect regional, crop-type, or language-based bias in AI recommendations
2. WHEN bias is detected above 10% variance in accuracy across regions or languages, THE KrishiSarathi SHALL trigger a model retraining process
3. THE KrishiSarathi SHALL maintain performance metrics disaggregated by region, language, and crop type for transparency
4. THE KrishiSarathi SHALL ensure training data includes representative samples from all 8 Regional_Language regions
5. WHEN new features are added, THE KrishiSarathi SHALL validate equal performance across all supported languages before deployment

### Requirement 11: Human Expert Escalation

**User Story:** As Ram Singh (farmer with a complex crop problem), I want access to human experts when the AI cannot confidently help me, so that I always have a path to reliable assistance.

#### Acceptance Criteria

1. WHEN the Disease_Diagnoser Confidence_Score is below 60%, THE KrishiSarathi SHALL automatically escalate to an Expert
2. WHEN a User explicitly requests human assistance, THE KrishiSarathi SHALL provide contact information for agricultural extension officers or helplines
3. WHEN escalating to an Expert, THE KrishiSarathi SHALL provide the Expert with full conversation history and AI analysis
4. THE KrishiSarathi SHALL maintain a network of at least 100 verified agricultural Experts across different regions and specializations
5. WHEN an Expert resolves a case, THE KrishiSarathi SHALL use that feedback to improve future AI recommendations

### Requirement 12: Privacy and Data Protection

**User Story:** As a User concerned about my personal information, I want my agricultural queries and personal details to be kept private and secure, so that I can use the system without fear of data misuse.

#### Acceptance Criteria

1. THE KrishiSarathi SHALL not store personally identifiable information (name, phone number, exact location) beyond the active session
2. WHEN collecting usage data for AI improvement, THE KrishiSarathi SHALL anonymize all data before storage
3. THE KrishiSarathi SHALL use location data only for providing relevant services and SHALL not share it with third parties
4. WHEN a User's session ends, THE KrishiSarathi SHALL delete all personal data within 24 hours except anonymized usage statistics
5. THE KrishiSarathi SHALL comply with Indian data protection regulations and provide Users with a clear privacy policy in their Regional_Language

### Requirement 13: Cost-Effective Operations

**User Story:** As a Government Officer managing agricultural technology budgets, I want the system to operate efficiently at scale, so that we can serve maximum farmers within budget constraints.

#### Acceptance Criteria

1. THE KrishiSarathi SHALL maintain operational costs below ₹5 per User per month at scale (100,000+ Users)
2. WHEN optimizing costs, THE KrishiSarathi SHALL prioritize AI model efficiency without compromising accuracy below 80%
3. THE KrishiSarathi SHALL use caching strategies to minimize repeated API calls for common queries
4. THE KrishiSarathi SHALL monitor and report cost metrics per feature (voice recognition, disease diagnosis, etc.) for optimization
5. WHEN costs exceed budget thresholds, THE KrishiSarathi SHALL alert administrators and suggest optimization strategies

### Requirement 14: Offline Capability for Critical Features

**User Story:** As Ram Singh (farmer in an area with intermittent connectivity), I want to access basic features even when my internet connection is unstable, so that I can get help during connectivity issues.

#### Acceptance Criteria

1. WHERE a User has the smartphone app installed, THE KrishiSarathi SHALL provide offline access to previously viewed disease information and treatment guides
2. WHEN connectivity is lost, THE KrishiSarathi SHALL queue User queries and process them when connection is restored
3. THE KrishiSarathi SHALL cache the most recent weather forecast and market prices for offline viewing
4. WHEN operating offline, THE KrishiSarathi SHALL clearly indicate which information is cached and when it was last updated
5. THE KrishiSarathi SHALL prioritize downloading essential content (disease guides, emergency contacts) for offline use

### Requirement 15: Continuous Learning and Improvement

**User Story:** As a Government Officer monitoring system effectiveness, I want the AI to continuously improve based on real-world usage and expert feedback, so that recommendations become more accurate over time.

#### Acceptance Criteria

1. WHEN an Expert corrects an AI diagnosis, THE KrishiSarathi SHALL incorporate that feedback into the training dataset
2. THE KrishiSarathi SHALL retrain AI models at least monthly using accumulated feedback and new agricultural research
3. WHEN model accuracy improves by more than 5%, THE KrishiSarathi SHALL deploy the updated model after validation testing
4. THE KrishiSarathi SHALL track accuracy metrics over time and report improvement trends to administrators
5. WHEN Users provide feedback on recommendation quality, THE KrishiSarathi SHALL use that data to identify areas for improvement
## AI Necessity Justification

### Requirement 1: Voice Interface
- **Rule-based limitation**: Cannot handle 200+ Indian dialects, accents
- **AI solution**: Fine-tuned Whisper model trained on farmer speech
- **Why AI needed**: Speech patterns vary by region, age, education level

### Requirement 2: Disease Diagnosis  
- **Rule-based limitation**: Static symptom-disease mapping (10,000+ combinations)
- **AI solution**: Neural network learns from 50,000 farmer cases
- **Why AI needed**: Same symptom = different disease based on weather/soil

## AI Model Performance Requirements

1. **Speech Recognition**: 
   - WER (Word Error Rate) < 15% for Hindi
   - WER < 20% for other regional languages
   - Inference time: < 2 seconds

2. **Disease Classifier**:
   - Precision: > 85% for top 20 crops
   - Recall: > 80% for common diseases
   - Confidence threshold for escalation: 60%

3. **NLP Intent Classification**:
   - Accuracy: > 90% for 7 intent categories
   - F1-score: > 0.85 across all languages

   ## Data Requirements

### Training Data Sources:
1. **Voice Data**: 100 hours per language (farmers speaking)
2. **Disease Dataset**: 10,000 labeled farmer queries (symptom → diagnosis)
3. **Agricultural Corpus**: 50,000 Q&A pairs in regional languages
4. **Government Data**: PM-KISAN, Soil Health Card, AgMarkNet APIs

### Data Collection Strategy:
- Partner with KVKs (Krishi Vigyan Kendras)
- Crowdsource from existing farmer WhatsApp groups
- Use government open data portals

## System Context Diagram
┌─────────────────────────────────────────────────┐
│ FARMER ECOSYSTEM │
│ [Small Farmer] [Marginal Farmer] [Landless] │
└──────────────────────┬──────────────────────────┘
│ Multiple Access Channels
┌──────────────────────▼──────────────────────────┐
│ KRISHISARATHI PLATFORM │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ Web │ │ WhatsApp │ │ IVR │ │
│ │ Interface│ │ Bot │ │ System │ │
│ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
└───────┼────────────┼─────────────┼─────────────┘
│ │ │
┌───────▼────────────▼─────────────▼─────────────┐
│ UNIFIED AI ENGINE CORE │
│ Speech → Text → Understanding → Knowledge → │
│ ↓ ↓ ↓ ↓ │
│ Response ← Synthesis ← Planning ← Retrieval │
└───────┬────────────┬─────────────┬─────────────┘
│ │ │
┌───────▼────────────▼─────────────▼─────────────┐
│ EXTERNAL SYSTEMS │
│ [IMD Weather] [AgMarkNet] [Govt Schemes] │
└────────────────────────────────────────────────┘

## Requirements to Design Traceability

| Requirement ID | Design Component |
|---------------|------------------|
| R1 – Voice Interface | Speech Service, NLU Service |
| R2 – Disease Diagnosis | Disease Service, Response Generator |
| R3 – Scheme Navigation | Scheme Service |
| R4 – Market Intelligence | Market Service, Redis Cache |
| R5 – Weather Alerts | Weather Service |
| R6 – Multi-channel | API Gateway, Interface Layer |
| R9 – Explainable AI | Response Generator, Analytics |
| R11 – Expert Escalation | Expert Service |
