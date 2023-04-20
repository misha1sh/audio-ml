'use strict';

const assert = require('chai').assert;
const fs = require('fs')

const file = fs.readFileSync('./model.onnx').buffer;

const ort = function() {
  if (typeof window === 'undefined') {
    return require('onnxruntime-node');
  }
  return require('onnxruntime-web');
}();


const session_promise = ort.InferenceSession.create(file, { executionProviders: ['cpu'] }); //, 0, file.byteLength,  { executionProviders: ['cpu'] });

module.exports.getPunctuation = async (text) => {
  assert.typeOf(text, 'string');

  let x = new Float32Array(32 * 489);
  for (let i = 0; i < x.length; i++) {
    x[i] = 1;
  }
  console.log(x);
  const tensor = new ort.Tensor('float32', x, [1 , 32, 489]);
  const session = await session_promise;
  const prom = session.run({input: tensor});
  const output = (await prom).output;
  assert.deepStrictEqual(output.dims, [1, 3]);


  console.log(output.data);

  return text;
}
