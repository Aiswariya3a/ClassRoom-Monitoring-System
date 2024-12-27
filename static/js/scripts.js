document.getElementById("capture-btn").addEventListener("click", () => {
  fetch("/capture_images", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        alert(`Error: ${data.error}`);
      } else {
        alert("Images captured successfully!");
      }
    });
});

document.getElementById("analyze-btn").addEventListener("click", () => {
  fetch("/analyze_images", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById(
        "engagement-result"
      ).innerText = `Average Engagement Score: ${data.average_engagement.toFixed(
        2
      )}`;
    });
});

// Fetch engagement data from the backend
fetch("../static/engagement_scores.json")
  .then((response) => response.json())
  .then((data) => {
    const engagementData = data;

    // Process session data for chart and list
    const sessionLabels = engagementData.map((item) => item.session);
    const engagementScores = engagementData.map((item) => item.score);

    // Display the last engagement score in a pie chart
    updateLastEngagementChart(engagementScores[engagementScores.length - 1]);

    // Update the overall score and display session history
    updateSessionHistory(engagementData);
    updateOverallScore(engagementScores);

    // Update the main engagement chart
    updateEngagementChart(sessionLabels, engagementScores);
  })
  .catch((error) => {
    console.error("Error fetching engagement data:", error);
    alert("Failed to load engagement data");
  });

// Function to update the pie chart for the last session's engagement score
function updateLastEngagementChart(lastScore) {
  const ctx = document.getElementById("lastEngagementChart").getContext("2d");
  new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Engagement", "Remaining"],
      datasets: [
        {
          data: [lastScore, 100 - lastScore],
          backgroundColor: [
            "rgba(75, 192, 192, 1)",
            "rgba(200, 200, 200, 0.3)",
          ],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          callbacks: {
            label: function (tooltipItem) {
              return tooltipItem.raw.toFixed(2) + "%";
            },
          },
        },
      },
    },
  });
}

// Function to update the session history chart
function updateEngagementChart(labels, scores) {
  const ctx = document.getElementById("engagementChart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Engagement Score (%)",
          data: scores,
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          borderColor: "rgba(75, 192, 192, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { stepSize: 10 },
        },
      },
    },
  });
}

// Function to update session history on the page
function updateSessionHistory(engagementData) {
  const slotList = document.getElementById("slot-list");
  slotList.innerHTML = ""; // Clear previous sessions

  engagementData.forEach((slot) => {
    const listItem = document.createElement("div");
    listItem.classList.add("slot-item");
    listItem.innerHTML = `
            <span> ${slot.session} | Score: <span class="score">${slot.score}</span>%</span>
        `;
    slotList.appendChild(listItem);
  });
}

// Function to calculate and display the overall engagement score
function updateOverallScore(scores) {
  const overallScore =
    scores.reduce((sum, score) => sum + score, 0) / scores.length;
  document.getElementById(
    "overall-score"
  ).textContent = `${overallScore.toFixed(2)}%`;
}

// Start session function
function startNewSession() {
  alert("Starting a new session...");
  // Redirect to session start (replace URL as needed)
  window.location.href = "/start-session";
}
