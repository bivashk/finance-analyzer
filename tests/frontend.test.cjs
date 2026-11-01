const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const sandbox = {document: {addEventListener() {}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('script.js','utf8')+'\nthis.Analyzer = FinanceAnalyzer;',sandbox);
const app = Object.create(sandbox.Analyzer.prototype);

test('uncertain suggestions remain for review; confident predictions are applied', async () => {
  app.state = {transactions:[{id:'abc',category:'Other'},{id:'def',category:'Other'}]};
  app.analyticsRequest = async () => ({predictions:[
    {category:'Rent',confidence:.4,needs_review:true},
    {category:'Income',confidence:.9,needs_review:false}]});
  app.showToast = app.renderTransactions = app.updateDashboard = app.populateCategoryFilter = app.saveState = () => {};
  await app.categorizeTransactions();
  assert.equal(app.state.transactions[0].category,'Other');
  assert.equal(app.state.transactions[0].prediction.category,'Rent');
  assert.equal(app.state.transactions[1].category,'Income');
});

test('server-only chat receives exactly the active transaction list', async () => {
  app.state = {transactions:[{id:'active'}]};
  app.analyticsRequest = async (path, body) => {
    assert.equal(path,'/api/chat');
    assert.equal(body.transactions,app.state.transactions);
    assert.equal(body.message,'expenses?');
    return {content:'verified'};
  };
  assert.equal(await app.callAPI('expenses?'),'verified');
});
