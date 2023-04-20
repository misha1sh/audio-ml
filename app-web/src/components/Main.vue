<template>
  <div class="hello">
    <h1>Исходный текст: </h1>
    <n-input
      type="textarea"
      v-model:value="input"
      placeholder="Исходный текст:"
      clearable
      :autosize="{
        minRows: 4
      }"
    />
    <n-button
      type="info"
      @click="handleInput"
      :disabled="showLoadingBar">Обработать</n-button>

    <h1>Результаты: </h1>

    <n-spin :show="showLoadingBar">
      <n-card  style="white-space: pre;">
        {{ results }}
      </n-card>

      <template #description>
        Loading python scripts
      </template>
    </n-spin>
  </div>
</template>

<script setup>
  import { NCard, NInput, NButton, NSpin } from 'naive-ui'
</script>

<script>
      // #<!-- @input="handleInput" -->
import {ref} from 'vue'

import { InferenceSession, env, Tensor } from 'onnxruntime-web';

env.wasm.wasmPaths = {
  'ort-wasm.wasm': "http://localhost:3000/js/ort-wasm.wasm",
  'ort-wasm-threaded.wasm':  "http://localhost:3000/js/ort-wasm-threaded.wasm",
  'ort-wasm-simd.wasm':  "http://localhost:3000/js/ort-wasm-simd.wasm",
  'ort-wasm-simd-threaded.wasm':  "http://localhost:3000/js/ort-wasm-simd-threaded.wasm",
}
// const model = require("../assets/model.onnx")
import buffer from "../assets/model.onnx";
const session_promise = InferenceSession.create(buffer,
    { executionProviders: ['wasm'] });


import { loadPyodide } from 'pyodide'
const pyodide_promise = loadPyodide({
    indexURL: "http://localhost:3000/pyodide/",
  });



import code from '../assets/code.py'
export default {
  name: 'HelloWorld',
  props: {
  },
  data() {
    return {
      input: "",
      results: "\n\n\n\n",
      showLoadingBar: ref(true)
    }
  },
  created() {
    async function run() {
      const pyodide = await pyodide_promise;
      await pyodide.loadPackage("micropip")
      const session = await session_promise;

      async function infer(float32_array) {
        console.log(float32_array)
        const tensor = new Tensor('float32', float32_array, [1 , 32, 489]);
        const prom = session.run({input: tensor});
        return (await prom).output;
      }

      await pyodide.registerJsModule("jsinfer", {
        "infer": infer
      })
      console.log(await pyodide.runPythonAsync(code))
      console.log(await pyodide.runPythonAsync("infer()", locals={
        text: "Тест.",
      }))
    }
    run().then(() => {
      this.showLoadingBar = false;
    })
    pyodide_promise.then(pyodide => {
      console.log(pyodide)

    })
  },
  methods: {
    async handleInput() {
      this.results = this.input;
      this.showLoadingBar = false;

      // let x = new Float32Array(32 * 489);
      // for (let i = 0; i < x.length; i++) {
      //   x[i] = 1;
      // }

      // console.log(output)

      // const pyodide = await pyodide_promise;
      // await pyodide.loadPackage("micropip")
      // console.log(await pyodide.runPythonAsync(code))

    }
  }
}
</script>

<style scoped>

</style>
