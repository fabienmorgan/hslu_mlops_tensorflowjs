console.warn = function () {};

const tf = require("@tensorflow/tfjs-node");
const { performance } = require("perf_hooks");
require("@tensorflow/tfjs-node"); // Use '@tensorflow/tfjs-node-gpu' if running with GPU.

const mnist = require("mnist"); // You need to install this package: 'yarn add mnist'
const set = mnist.set(8000, 2000); // 8000 for training, 2000 for testing.

if (tf.getBackend() === "cpu") {
  console.log("Training runs on GPU.");
} else {
  console.log("Training runs on CPU.");
}

const trainingSet = set.training;
console.log("Number of training set: ", trainingSet.length);

const testingSet = set.test;

// Prepare the data
const trainData = {
  size: trainingSet.length,
  input: tf.reshape(tf.tensor2d(trainingSet.map((item) => item.input)), [
    trainingSet.length,
    28,
    28,
    1,
  ]),
  output: tf.tensor2d(trainingSet.map((item) => item.output)),
};

const testData = {
  size: testingSet.length,
  input: tf.reshape(tf.tensor2d(testingSet.map((item) => item.input)), [
    testingSet.length,
    28,
    28,
    1,
  ]),
  output: tf.tensor2d(testingSet.map((item) => item.output)),
};
// Define the model
const model = tf.sequential();
model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
model.add(tf.layers.dense({ units: 100, activation: "relu" }));
model.add(tf.layers.dense({ units: 100, activation: "relu" }));
model.add(tf.layers.dense({ units: 100, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

// Compile the model
model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

model.summary();

// Train the model
async function train() {
  const startTime = performance.now();
  await model.fit(trainData.input, trainData.output, {
    epochs: 50,
    validationSplit: 0.2, // Added validation split of 0.2
  });
  const endTime = performance.now();
  console.log(`Training time: ${(endTime - startTime) / 1000} seconds`);
}
train();
