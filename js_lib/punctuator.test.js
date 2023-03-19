'use strict';

const getPunctuation = require('./').getPunctuation;


test('1', async () => {
    expect(await getPunctuation('Тест')).toBe('Тест.');
});