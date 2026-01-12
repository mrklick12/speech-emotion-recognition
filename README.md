<h2>Machine Learning Speech Emotion Recognition</h2>

<h6>Speech emotion recognition system using classical ML and engineered audio features.</h6>



<img width="2535" height="731" alt="image" src="https://github.com/user-attachments/assets/7aa044d0-0238-4b13-837e-3e63a7582b40" />

<h3>Motivation</h3>

The human speech carries far more information than words alone. Emotion, intent, and emphasis are embedded in acoustic patterns such as pitch, energy, and rhythm. These can simply be numerical infomation when extracted from an audio file.

As someone interested in machine learning and artifiical inteligence, I wanted to explore how much of this information can be captured using practical, interpretable methods and how we can interpret that into a more qualitative aspect.

This project was driven by the idea that emotionally aware systems are key to making human–machine interaction feel natural rather than mechanical. Voice is one of the richest carriers of emotion, and work like that done at ElevenLabs shows how expressive and human-centred speech technology can be.

<h3>What I Built</h3>
<p>
  I implemented a complete speech emotion recognition (SER) pipeline using the <b>RAVDESS</b> dataset as a benchmark.
  The system operates in four stages:       
</p>
<ol>
  <li>
    <b>Feature extraction</b>
    Hand-engineered audio features were extracted from raw speech, including:
    <ul>
      <li>Pitch statistics</li>
      <li>Energy (RMS)</li>
      <li>MFCCs (Mel-Frequency Cepstral Coefficients)</li>
      <li>Spectral features</li>
      <li>Temporal features such as onset rate and tempo</li>
    </ul>
    These features were chosen to reflect known acoustic correlates of emotion while remaining interpretable.
  </li>
  <li>
    <b>Model</b>
    The classifier is a <b>Support Vector Machine (SVC), implemented as a scikit-learn pipeline with
    preprocessing and imputation. This choice prioritises strong performance on limited data and
    robustness over architectural complexity.
  </li>
  <li>
    <b>Training and evaluation</b>
    On benchmark data, the model achieves <b>~80%+ accuracy</b> for multi-class emotion classification under standard
    train/test splits.
  </li>
  <li>
    <b>End-to-end inference</b><br/>
    A lightweight inference script loads a trained model, processes extracted features, and outputs predicted emotion
    labels, demonstrating a complete and reusable workflow.
  </li>
</ol>

<h3>Results and Limitations</h3>
<p>
  The model performs strongly <b>in-distribution</b>, where speaker characteristics and recording conditions are similar
  to the training data. As expected, performance drops when evaluated on out-of-distribution inputs (e.g. new speaker), 
  highlighting the challenge of domain shift in speech emotion recognition.
</p>
<p>
  Rather than treating this as a failure, I thought it reflects a realistic constraint of supervised ML systems and reinforces the
  importance of data diversity in real-world applications.
</p>


<h3>Why This Matters</h3>
<b>What I took from this project:</b>
<ul>
  <li>A solid understanding of <b>audio-based machine learning</b></li>
  <li>The ability to design and implement <b>end-to-end ML systems</b></li>
  <li>An appreciation of <b>engineering trade-offs and limitations</b>, not just accuracy metrics</li>
</ul>
<p>
  Most importantly, it reflects a genuine interest in how machine learning can be applied thoughtfully to human
  communication, an area I am keen to continue exploring further.
</p>
