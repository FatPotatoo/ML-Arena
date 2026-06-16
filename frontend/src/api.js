// One place that knows where the backend lives and how to call it.
// Keeping fetch calls here (not scattered in components) means if the address
// ever changes, we edit one line.
const API_BASE = 'http://localhost:8000'

// Ask the backend for the dataset summary. `async` means this returns a Promise
// (a value that arrives later, since the network takes time). `await` pauses
// until the response comes back.
export async function getDatasetInfo() {
  const response = await fetch(`${API_BASE}/api/dataset/info`)
  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`)
  }
  return response.json() // parse the JSON body into a JavaScript object
}

// Send the config and train a model. This is a POST (we're sending a body) —
// unlike the GET above which just asks for data.
export async function trainModel(config) {
  const response = await fetch(`${API_BASE}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config), // turn the JS config object into JSON text
  })
  const data = await response.json()
  if (!response.ok) {
    // FastAPI returns the reason in `detail` (a string for 400, a list for 422).
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    throw new Error(detail || `Request failed: ${response.status}`)
  }
  return data
}

