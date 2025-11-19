async function uploadFile() {
  const fileInput = document.getElementById("fileInput")
  if (!fileInput.files.length) return alert("choose a file first!")
  const formData = new FormData()
  formData.append("file", fileInput.files[0])

  const res = await fetch("/api/upload", { method: "POST", body: formData })
  const data = await res.json()

  const messages = document.getElementById("messages")
  const notification = document.createElement("div")
  notification.className = "upload-notification"
  notification.innerHTML = `<div class="notification-message">${
    data.message || data.error
  }</div>`
  messages.appendChild(notification)
  scrollToBottom()
}

async function sendQuery() {
  const queryInput = document.getElementById("query")
  const sendBtn = document.querySelector(".chat-input button")
  const query = queryInput.value.trim()
  if (!query) return

  queryInput.disabled = true
  sendBtn.disabled = true

  const messages = document.getElementById("messages")

  const userMsg = document.createElement("div")
  userMsg.className = "user"
  userMsg.innerHTML = `<div class="user-message">${escapeHtml(query)}</div>`
  messages.appendChild(userMsg)

  queryInput.value = ""

  const loadingContainer = document.createElement("div")
  loadingContainer.className = "loading-container"
  loadingContainer.innerHTML = `
    <div class="loading-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `
  messages.appendChild(loadingContainer)
  scrollToBottom()

  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  })

  const data = await res.json()

  loadingContainer.remove()

  let answer = data.answer || data.error

  answer = answer
    .replace(/\*\*\*([\s\S]+?)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*([\s\S]+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n\* /g, "\n• ")
    .replace(/\n(\d+)\. /g, "\n$1. ")
    .replace(/\n/g, "<br>")

  const botMsg = document.createElement("div")
  botMsg.className = "bot"
  botMsg.innerHTML = `<div class="bot-message">${answer}</div>`
  messages.appendChild(botMsg)
  scrollToBottom()

  queryInput.disabled = false
  sendBtn.disabled = false
  queryInput.focus()
}

function handleEnter(event) {
  if (event.key === "Enter") {
    event.preventDefault()
    sendQuery()
  }
}

function escapeHtml(text) {
  const div = document.createElement("div")
  div.textContent = text
  return div.innerHTML
}

function scrollToBottom() {
  const messages = document.getElementById("messages")
  messages.scrollTop = messages.scrollHeight
}
