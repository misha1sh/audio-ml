'use strict';

const assert = require('chai').assert;

const ort = require('onnxruntime-web');
const session_promise = ort.InferenceSession.create('../model.v1.onnx');

module.exports.getPunctuation = async (text) => {
  assert.typeOf(text, 'string');

  let x = new Float32Array(16 * 363);
  for (let i = 0; i < x.length; i++) {
    x[i] = 1;
  }

  const tensor = new ort.Tensor('float32', x, [1, 16, 363]);
  const session = await session_promise;
  const output = (await session.run({'inputt': tensor})).output;
  assert.deepStrictEqual(output.dims, [1, 3]);


  console.log(output.data);

  return text;
}
