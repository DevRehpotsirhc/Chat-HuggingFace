console.log('the js is actually running');

async function sendMessage() {
    const input = document.getElementById('inputMessage');
    const responseDiv = document.getElementById('response');
    const message = input.value;
    
    if (!message) {
        alert('Please type something!');
        return;
    }
    
    try {
        const response = await fetch('/api/archrouter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message
            })
        });
        
        const data = await response.json();
        console.log(data);
        
        if (data.route) {
            const ruta = data.route
            console.log(ruta);
            
            responseDiv.innerHTML = `<strong>Response:</strong> ${ruta}`;
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