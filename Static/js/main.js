function setupPreview(dropId, inputId, previewId) {
    const dropZone = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    dropZone.addEventListener('click', () => input.click());

    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                preview.src = event.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
}

setupPreview('dropRef', 'refImg', 'prevRef');
setupPreview('dropTest', 'testImg', 'prevTest');

document.getElementById('verifyForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btnVerify');
    const resultCard = document.getElementById('resultCard');
    const formData = new FormData(e.target);

    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch('/verify', { method: 'POST', body: formData });
        const data = await res.json();

        if (res.ok) {
            resultCard.classList.remove('hidden');
            const badge = document.getElementById('statusBadge');
            badge.innerText = data.status;
            badge.className = `status-badge ${data.match ? 'genuine' : 'forgery'}`;

            document.getElementById('similarityScore').innerText = `${data.similarity}%`;
            document.getElementById('distanceVal').innerText = data.distance;
        } else {
            alert(data.error || 'Verification failed');
        }
    } catch (err) {
        alert('An error occurred during verification.');
    } finally {
        btn.innerText = 'Verify Signatures';
        btn.disabled = false;
    }
});