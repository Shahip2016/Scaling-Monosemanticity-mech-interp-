/**
 * Demo data for the Scaling Monosemanticity dashboard.
 * Provides realistic mock data so the frontend works without a trained SAE.
 */

const DEMO_DATA = {
  /**
   * Feature analysis data — simulates output from analysis.py
   */
  features: {
    42: {
      feature_idx: 42,
      mean_activation: 0.0347,
      max_activation: 8.214,
      fraction_active: 0.0182,
      mean_activation_when_active: 1.907,
      top_examples: [
        {
          token_position: 14203,
          activation: 8.214,
          context_text: "The most famous suspension bridge in the world, the <span class='highlight'>Golden Gate Bridge</span>, spans the strait connecting San Francisco Bay to the Pacific Ocean.",
        },
        {
          token_position: 8901,
          activation: 7.453,
          context_text: "Crossing the <span class='highlight'>bridge</span> at sunset offers breathtaking views of the bay, Alcatraz Island, and the Marin Headlands.",
        },
        {
          token_position: 22107,
          activation: 6.891,
          context_text: "The distinctive International Orange color of the <span class='highlight'>Golden Gate</span> was chosen to enhance visibility in the fog.",
        },
        {
          token_position: 5529,
          activation: 6.102,
          context_text: "Construction of the <span class='highlight'>bridge</span> began in 1933 and was completed in April 1937, opening to vehicular traffic on May 28.",
        },
        {
          token_position: 31042,
          activation: 5.773,
          context_text: "San Francisco's iconic <span class='highlight'>bridge</span> stretches 1.7 miles across the Golden Gate strait, with towers reaching 746 feet.",
        },
        {
          token_position: 17865,
          activation: 5.341,
          context_text: "The <span class='highlight'>suspension</span> cables contain 80,000 miles of wire, enough to encircle the Earth three times.",
        },
        {
          token_position: 9244,
          activation: 4.892,
          context_text: "Joseph Strauss served as chief engineer of the <span class='highlight'>Golden Gate Bridge</span> project, though Irving Morrow designed its Art Deco styling.",
        },
        {
          token_position: 28713,
          activation: 4.201,
          context_text: "On foggy mornings, only the towers of the <span class='highlight'>bridge</span> are visible above the cloud layer, creating an ethereal scene.",
        },
      ],
    },
    137: {
      feature_idx: 137,
      mean_activation: 0.0215,
      max_activation: 6.77,
      fraction_active: 0.0098,
      mean_activation_when_active: 2.193,
      top_examples: [
        {
          token_position: 2044,
          activation: 6.77,
          context_text: "The <span class='highlight'>neural network</span> processes input data through multiple layers of interconnected nodes.",
        },
        {
          token_position: 11233,
          activation: 5.92,
          context_text: "Training a deep <span class='highlight'>neural</span> model requires large datasets and significant computational resources.",
        },
        {
          token_position: 7891,
          activation: 5.41,
          context_text: "Recent advances in <span class='highlight'>deep learning</span> have enabled breakthroughs in computer vision and NLP.",
        },
        {
          token_position: 19502,
          activation: 4.88,
          context_text: "The <span class='highlight'>transformer</span> architecture has become the dominant paradigm for language modeling tasks.",
        },
        {
          token_position: 25611,
          activation: 4.32,
          context_text: "Gradient <span class='highlight'>backpropagation</span> allows the network to learn by adjusting weights based on error signals.",
        },
      ],
    },
    256: {
      feature_idx: 256,
      mean_activation: 0.0089,
      max_activation: 4.21,
      fraction_active: 0.0043,
      mean_activation_when_active: 2.07,
      top_examples: [
        {
          token_position: 3201,
          activation: 4.21,
          context_text: "The president signed the <span class='highlight'>legislation</span> into law after months of congressional debate.",
        },
        {
          token_position: 15888,
          activation: 3.89,
          context_text: "The <span class='highlight'>Senate</span> voted 62-38 to approve the bipartisan infrastructure bill.",
        },
        {
          token_position: 9402,
          activation: 3.45,
          context_text: "Political analysts predict the upcoming <span class='highlight'>election</span> will be one of the most contested in decades.",
        },
      ],
    },
  },

  /**
   * Scaling laws sweep results — simulates output from scaling_laws.py
   */
  scalingLaws: {
    sweep: [
      { n_features: 4096, n_steps: 1000, loss: 0.482, flops: 2.01e12, l0: 85.2, variance_explained: 0.621 },
      { n_features: 4096, n_steps: 2500, loss: 0.391, flops: 5.03e12, l0: 112.4, variance_explained: 0.689 },
      { n_features: 4096, n_steps: 5000, loss: 0.342, flops: 1.01e13, l0: 128.7, variance_explained: 0.718 },
      { n_features: 16384, n_steps: 1000, loss: 0.398, flops: 8.05e12, l0: 145.1, variance_explained: 0.695 },
      { n_features: 16384, n_steps: 2500, loss: 0.301, flops: 2.01e13, l0: 198.3, variance_explained: 0.752 },
      { n_features: 16384, n_steps: 5000, loss: 0.258, flops: 4.03e13, l0: 221.6, variance_explained: 0.789 },
      { n_features: 65536, n_steps: 1000, loss: 0.351, flops: 3.22e13, l0: 201.4, variance_explained: 0.731 },
      { n_features: 65536, n_steps: 2500, loss: 0.247, flops: 8.05e13, l0: 278.9, variance_explained: 0.811 },
      { n_features: 65536, n_steps: 5000, loss: 0.198, flops: 1.61e14, l0: 294.2, variance_explained: 0.847 },
    ],
    allocation: {
      loss_power_law: { a: 1842.5, b: -0.162 },
      features_power_law: { a: 0.0043, b: 0.412 },
      steps_power_law: { a: 0.018, b: 0.287 },
    },
  },

  /**
   * Training history — simulates train_summary.json
   */
  trainingHistory: {
    config: {
      n_features: 16384,
      d_in: 768,
      l1_coefficient: 5.0,
      learning_rate: 3e-4,
      batch_size: 4096,
      n_steps: 10000,
    },
    history: (() => {
      const steps = [];
      for (let i = 1; i <= 100; i++) {
        const step = i * 100;
        const progress = step / 10000;
        steps.push({
          step,
          loss: 0.95 * Math.exp(-2.8 * progress) + 0.22 + (Math.random() - 0.5) * 0.015,
          mse: 0.42 * Math.exp(-3.1 * progress) + 0.12 + (Math.random() - 0.5) * 0.008,
          l1: 0.035 * Math.exp(-1.5 * progress) + 0.008 + (Math.random() - 0.5) * 0.002,
          l0: 45 + 180 * (1 - Math.exp(-3.5 * progress)) + (Math.random() - 0.5) * 10,
          variance_explained: 0.15 + 0.65 * (1 - Math.exp(-4.0 * progress)) + (Math.random() - 0.5) * 0.02,
          elapsed_s: step * 0.38,
        });
      }
      return steps;
    })(),
    summary: {
      dead_feature_fraction: 0.142,
      n_dead_features: 2326,
      final_loss: 0.258,
      final_variance_explained: 0.789,
      final_l0: 221.6,
    },
  },

  /**
   * Steering examples — simulates steering.py output
   */
  steering: {
    "Golden Gate Bridge": {
      prompt: "The famous landmark in San Francisco is the",
      baseline: "The famous landmark in San Francisco is the Golden Gate Park, which is a sprawling urban park that covers over 1,000 acres of land.",
      results: {
        "-30": "The famous landmark in San Francisco is the city's financial district, home to numerous technology companies and startups that drive the local economy.",
        "30": "The famous landmark in San Francisco is the Golden Gate Bridge, the magnificent suspension bridge that spans the Golden Gate strait with its iconic International Orange towers.",
        "60": "The famous landmark in San Francisco is the Golden Gate Bridge Golden Gate Bridge Golden Gate Bridge! The stunning bridge, the bridge, the incredible bridge spanning the bay.",
      },
    },
    "Neural Networks": {
      prompt: "The most important concept in machine learning is",
      baseline: "The most important concept in machine learning is the ability to learn from data without being explicitly programmed for every task.",
      results: {
        "-20": "The most important concept in machine learning is the careful selection of training data and proper feature engineering.",
        "20": "The most important concept in machine learning is the neural network, a computational model inspired by biological neural circuits in the brain.",
        "50": "The most important concept in machine learning is the deep neural network architecture, with layers of neurons performing backpropagation and gradient descent optimization.",
      },
    },
  },

  /**
   * Available feature indices for the explorer dropdown
   */
  availableFeatures: [42, 137, 256, 512, 1024, 2048, 4096, 8192],
};

// Make globally available
if (typeof window !== "undefined") {
  window.DEMO_DATA = DEMO_DATA;
}
