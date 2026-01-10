<h2>Machine Learning Speech Emotion Recognition</h2>

<h6>High-accuracy speech emotion recognition system using classical ML and engineered audio features.</h6>



<img width="2535" height="731" alt="image" src="https://github.com/user-attachments/assets/7aa044d0-0238-4b13-837e-3e63a7582b40" />

<b>Motivation</b>

Human speech carries far more information than words alone. Emotion, intent, and emphasis are embedded in acoustic patterns such as pitch, energy, and rhythm. 

As someone interested in machine learning and artifiical inteligence, I wanted to explore how much of this information can be captured using practical, interpretable methods and how we can interpret that into a more qualitative aspect.

This project was motivated by the idea that emotionally aware systems are a key step towards more natural and human-like interaction between people and machines. Instead of chasing novelty or over-complexity, the focus here is on building a system that works end-to-end, is understandable, and highlights both the strengths and limits of classical ML when applied to real audio data.

<h3>What I Built</h3>
<p>
  I implemented a complete speech emotion recognition (SER) pipeline using the <b>RAVDESS</b> dataset as a benchmark.
  The system operates in four stages:
</p>
<ol>
  <li>
    <b>Feature extraction</b><br/>
    Hand-engineered audio features were extracted from raw speech, including:
    <ul>
      <li>Pitch statistics</li>
      <li>Energy (RMS)</li>
      <li>MFCCs</li>
      <li>Spectral features</li>
      <li>Temporal features such as onset rate and tempo</li>
    </ul>
    These features were chosen to reflect known acoustic correlates of emotion while remaining interpretable.
  </li>
  <li>
    <b>Model</b><br/>
    The classifier is a <b>Support Vector Machine (SVC)</b> with an <b>RBF kernel</b>, implemented as a scikit-learn
    pipeline with preprocessing and imputation. This choice prioritises strong performance on limited data and
    robustness over architectural complexity.
  </li>
  <li>
    <b>Training and evaluation</b><br/>
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
  to the training data. As expected, performance drops when evaluated on out-of-distribution inputs (e.g. new speakers
  or recording setups), highlighting the challenge of domain shift in speech emotion recognition.
</p>
<p>
  Rather than treating this as a failure, it reflects a realistic constraint of supervised ML systems and reinforces the
  importance of speaker normalisation, data diversity, and domain adaptation in real-world deployments.
</p>

<h3>Demo</h3>
<p>
  Inference can be run in a few lines by loading the trained pipeline and passing extracted features into the model. A
  minimal example is included in the repository to demonstrate this process clearly.
</p>

<h3>Why This Matters</h3>
<ul>
  <li>A solid understanding of <b>audio-based machine learning</b></li>
  <li>The ability to design and implement <b>end-to-end ML systems</b></li>
  <li>An appreciation of <b>engineering trade-offs and limitations</b>, not just accuracy metrics</li>
</ul>
<p>
  Most importantly, it reflects a genuine interest in how machine learning can be applied thoughtfully to human
  communication — an area I am keen to continue exploring further.
</p>
