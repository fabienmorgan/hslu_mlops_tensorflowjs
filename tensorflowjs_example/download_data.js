const fs = require("fs");
const mnist = require("mnist");
const set = mnist.set(8000, 2000);

function reshape(input) {
  let output = [];
  for (let i = 0; i < 28; i++) {
    output[i] = [];
    for (let j = 0; j < 28; j++) {
      output[i][j] = input[i * 28 + j];
    }
  }
  return output;
}

const data = {
  training: set.training.map((sample) => ({
    input: reshape(sample.input),
    output: sample.output,
  })),
  test: set.test.map((sample) => ({
    input: reshape(sample.input),
    output: sample.output,
  })),
};

fs.writeFileSync("mnist.json", JSON.stringify(data));
