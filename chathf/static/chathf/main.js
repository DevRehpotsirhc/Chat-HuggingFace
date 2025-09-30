console.log('the js is actually running');

async function sendMessage() {
    const input = document.getElementById('inputMessage');
    const file = document.getElementById('fileInput').files[0];
    const responseDiv = document.getElementById('response');
    const message = input.value;
    
    if (!message) {
        alert('Please type something!');
        return;
    }

    const formData = new FormData();
    formData.append('message', message)
    
    if (file) {
        formData.append('file', file);
    }

    try {
        const response = await fetch('/api/archrouter', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        console.log(data);
        
        if (data.answer) {
            const respuesta = data.answer
            console.log(respuesta);
            
            responseDiv.innerHTML = `<strong>Response:</strong> ${respuesta}`;
            responseDiv.style.backgroundColor = '#d4edda';
        } else {
            responseDiv.innerHTML = `<strong>Error:</strong> ${data.error}`;
            responseDiv.style.backgroundColor = '#f8d7da';
        }
        
        input.value = '';
        
    } catch (error) {
        responseDiv.innerHTML = `<strong>Network Error:</strong> ${error.message}`;
        responseDiv.style.backgroundColor = '#f8d7da';
    }
}

document.getElementById('inputMessage').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});