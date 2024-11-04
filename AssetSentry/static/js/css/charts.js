let timeSeriesChart, anomalyScoreChart;
const maxDataPoints = 50;

function initCharts() {
    const timeSeriesCtx = document.getElementById('timeSeriesChart').getContext('2d');
    timeSeriesChart = new Chart(timeSeriesCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Time Series Prediction',
                data: [],
                borderColor: 'blue',
                fill: false
            }]
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: 'Time Series Prediction'
            }
        }
    });

    const anomalyScoreCtx = document.getElementById('anomalyScoreChart').getContext('2d');
    anomalyScoreChart = new Chart(anomalyScoreCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Anomaly Score',
                data: [],
                borderColor: 'red',
                fill: false
            }]
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: 'Anomaly Score'
            }
        }
    });
}

function updateCharts() {
    fetch('/predict')
        .then(response => response.json())
        .then(data => {
            const timestamp = new Date().toLocaleTimeString();

            // Update Time Series Chart
            timeSeriesChart.data.labels.push(timestamp);
            timeSeriesChart.data.datasets[0].data.push(data.prediction);
            if (timeSeriesChart.data.labels.length > maxDataPoints) {
                timeSeriesChart.data.labels.shift();
                timeSeriesChart.data.datasets[0].data.shift();
            }
            timeSeriesChart.update();

            // Update Anomaly Score Chart
            anomalyScoreChart.data.labels.push(timestamp);
            anomalyScoreChart.data.datasets[0].data.push(data.anomaly_score);
            if (anomalyScoreChart.data.labels.length > maxDataPoints) {
                anomalyScoreChart.data.labels.shift();
                anomalyScoreChart.data.datasets[0].data.shift();
            }
            anomalyScoreChart.update();
        });
}

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setInterval(updateCharts, 5000); // Update every 5 seconds
});