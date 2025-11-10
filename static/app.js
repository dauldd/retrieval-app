async function uploadFile() {
  const fileInput = document.getElementById("fileInput")
  if (!fileInput.files.length) return alert("choose a file first!")
  const formData = new FormData()
  formData.append("file", fileInput.files[0])

  const res = await fetch("/api/upload", { method: "POST", body: formData })
  const data = await res.json()
  document.getElementById("uploadStatus").innerText = data.message || data.error
}

async function sendQuery() {
  const query = document.getElementById("query").value
  const messages = document.getElementById("messages")
  messages.innerHTML += `<div class='user'>${query}</div>`

  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  })

  const data = await res.json()
  messages.innerHTML += `<div class='bot'>${data.answer || data.error}</div>`
}
