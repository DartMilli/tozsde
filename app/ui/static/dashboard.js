// === Ábrák és indikátorok betöltése ===
function showChart(imageUrl, clickedElement) {
    const imageElement = document.getElementById('chart-image');
    const descriptionElement = document.getElementById('chart-description');
    const loadingSpinner = document.getElementById('loading-spinner');
    const indicatorContainer = document.getElementById('indicator-table-container');

    imageElement.src = '';
    imageElement.style.display = 'none';
    loadingSpinner.style.display = 'block';
    descriptionElement.innerText = '';
    indicatorContainer.innerHTML = '';

    document.querySelectorAll('.chart-option').forEach(el => el.classList.remove('active'));
    clickedElement.classList.add('active');
    descriptionElement.innerText = clickedElement.getAttribute("info");

    const start = daysToDate(parseInt(sliderStart.value));
    const end = daysToDate(parseInt(sliderEnd.value));
    const urlWithDates = `${imageUrl}&start=${start}&end=${end}`;
    imageElement.src = urlWithDates;

    imageElement.onload = async () => {
        loadingSpinner.style.display = 'none';
        imageElement.style.display = 'block';

        // paramétereket az URL-ből
        const urlParams = new URLSearchParams(imageElement.src.split('?')[1]);
        const ticker = urlParams.get("ticker");
        const start = urlParams.get("start");
        const end = urlParams.get("end");

        // === Riport gomb beállítása ===
        const reportButtonContainer = document.getElementById('report-button-container');
        const reportButton = document.getElementById('view-report-button');

        const reportUrl = `/report?ticker=${ticker}&start=${start}&end=${end}`;
        reportButton.href = reportUrl;

        reportButtonContainer.style.display = 'block';

        // === Indikátor leírások betöltése ===
        try {
            const response = await fetch('/indicators');
            const data = await response.json();

            for (const key in data) {
                const ind = data[key];
                const id = 'table-' + key.toLowerCase();
                const section = document.createElement('div');
                section.classList.add('indicator-section');

                section.innerHTML = `
                    <h3 id="header-${id}" onclick="toggleTable('${id}')">
                        ${key + " - " + ind.name}
                        <span class="toggle-icon" id="icon-${id}">▶</span>
                    </h3>
                    <table id="${id}" class="indicator-table">
                        <tr><th>Amit mutat</th><td>${ind.shows}</td></tr>
                        <tr><th>Használat</th><td>${ind.used_for.map(i => `• ${i}`).join('<br>')}</td></tr>
                        <tr><th>Szignálok</th><td>${ind.signals.map(i => `• ${i}`).join('<br>')}</td></tr>
                        <tr><th>Erősségek</th><td>${ind.strengths.map(i => `✓ ${i}`).join('<br>')}</td></tr>
                        <tr><th>Gyengeségek</th><td>${ind.weaknesses.map(i => `✗ ${i}`).join('<br>')}</td></tr>
                    </table>
                `;
                indicatorContainer.appendChild(section);
            }
        } catch (error) {
            indicatorContainer.innerHTML = "<p>Nem sikerült betölteni az indikátorokat.</p>";
        }

        // === Ajánlás történet lekérése ===
        const historySection = document.getElementById("history-section");
        const historyTableBody = document.querySelector("#history-table tbody");
        const historyTitle = document.getElementById("history-title");

        try {
            const res = await fetch(`/history?ticker=${ticker}&start=${start}&end=${end}`);
            const json = await res.json();

            if (json.history && json.history.length > 0) {
                historyTableBody.innerHTML = "";
                json.history.forEach(row => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td>${row.date}</td>
                        <td>${row.recommendation}</td>
                        <td>${row.confidence.toFixed(2)}</td>
                    `;
                    historyTableBody.appendChild(tr);
                });

                historyTitle.textContent = `${ticker} - Ajánlások (${start} – ${end})`;
                historySection.style.display = "block";
            } else {
                historySection.style.display = "none";
            }
        } catch (error) {
            console.error("Ajánlás történet betöltése sikertelen:", error);
            historySection.style.display = "none";
        }
    };

    imageElement.onerror = () => {
        loadingSpinner.style.display = 'none';
        imageElement.src = '';
        imageElement.alt = 'Grafikon betöltése sikertelen.';
        imageElement.style.display = 'block';
    };
}

function toggleTable(id) {
    const table = document.getElementById(id);
    const icon = document.getElementById('icon-' + id);
    const header = document.getElementById('header-' + id);

    if (table.style.display === 'none' || table.style.display === '') {
        table.style.display = 'table';
        icon.textContent = '▼';
        header.classList.remove('collapsed');
    } else {
        table.style.display = 'none';
        icon.textContent = '▶';
        header.classList.add('collapsed');
    }
}

const baseDate = new Date("2020-01-01");
const today = new Date();
const daysDiff = Math.floor((today - baseDate) / (1000 * 60 * 60 * 24));

const sliderStart = document.getElementById("range-start");
const sliderEnd = document.getElementById("range-end");
const labelStart = document.getElementById("range-start-label");
const labelEnd = document.getElementById("range-end-label");
const diffLabel = document.getElementById("range-diff-label");
const highlight = document.querySelector('.slider-track-highlight');

const minGapDays = 91;

sliderStart.min = 0;
sliderStart.max = daysDiff;
sliderEnd.min = 0;
sliderEnd.max = daysDiff;
sliderStart.step = 1;
sliderEnd.step = 1;

sliderEnd.value = daysDiff;
sliderStart.value = Math.max(daysDiff - 182, 0);

function daysToDate(dayIndex) {
    const date = new Date(baseDate);
    date.setDate(date.getDate() + parseInt(dayIndex));
    return date.toISOString().slice(0, 10);
}

function daysToDateObject(dayIndex) {
    const date = new Date(baseDate);
    date.setDate(date.getDate() + parseInt(dayIndex));
    return date;
}

function updateSliderUI(movedSlider) {
    let startVal = parseInt(sliderStart.value);
    let endVal = parseInt(sliderEnd.value);

    if (movedSlider === sliderStart) {
        if (startVal > endVal - minGapDays) {
            endVal = startVal + minGapDays;
            if (endVal > daysDiff) {
                endVal = daysDiff;
                startVal = endVal - minGapDays;
            }
        }
    } else {
        if (endVal < startVal + minGapDays) {
            startVal = endVal - minGapDays;
            if (startVal < 0) {
                startVal = 0;
                endVal = startVal + minGapDays;
            }
        }
    }

    sliderStart.value = startVal;
    sliderEnd.value = endVal;

    const startDate = daysToDateObject(startVal);
    const endDate = daysToDateObject(endVal);
    labelStart.textContent = startDate.toISOString().slice(0, 10);
    labelEnd.textContent = endDate.toISOString().slice(0, 10);

    let years = endDate.getFullYear() - startDate.getFullYear();
    let months = endDate.getMonth() - startDate.getMonth();
    let days = endDate.getDate() - startDate.getDate();
    if (days < 0) {
        months--;
        const prevMonthLastDay = new Date(endDate.getFullYear(), endDate.getMonth(), 0).getDate();
        days += prevMonthLastDay;
    }
    if (months < 0) {
        years--;
        months += 12;
    }
    const totalMonths = years * 12 + months;
    diffLabel.textContent = `(${totalMonths} hónap, ${days} nap)`;

    const totalRange = sliderStart.max - sliderStart.min;
    const startPercent = (startVal / totalRange) * 100;
    const endPercent = (endVal / totalRange) * 100;
    highlight.style.left = `${startPercent}%`;
    highlight.style.width = `${endPercent - startPercent}%`;
}

sliderStart.addEventListener('input', () => updateSliderUI(sliderStart));
sliderEnd.addEventListener('input', () => updateSliderUI(sliderEnd));

updateSliderUI();
