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
