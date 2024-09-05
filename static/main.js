document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(e) {
        const textarea = document.querySelector('textarea');
        if (textarea.value.trim() === "") {
            e.preventDefault();
            alert("Please enter a message.");
        }
    });
});
