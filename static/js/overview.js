// Animation timeline for Grok-3+ architecture overview
document.addEventListener('DOMContentLoaded', function() {
    // Initialize cards
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach(card => {
        card.style.opacity = 0;
        card.style.transform = 'translateY(20px)';
    });
    
    // Start the animation sequence
    startAnimation();
    
    // Auto-loop every 8 seconds
    setInterval(() => {
        replayAnimation();
    }, 8000);
    
    // Add click event to replay button
    document.getElementById('replay-btn').addEventListener('click', replayAnimation);
});

// Main animation timeline
function startAnimation() {
    const timeline = anime.timeline({
        easing: 'easeOutExpo',
        duration: 800
    });
    
    // Animate title
    timeline.add({
        targets: '.hero-title',
        opacity: [0, 1],
        translateY: [20, 0],
    });
    
    // Animate subtitle
    timeline.add({
        targets: '.hero-subtitle',
        opacity: [0, 1],
        translateY: [20, 0],
    }, '-=600');
    
    // Animate cards one by one
    const cards = document.querySelectorAll('.feature-card');
    cards.forEach((card, index) => {
        timeline.add({
            targets: card,
            opacity: [0, 1],
            translateY: [20, 0],
            delay: anime.stagger(100)
        }, '-=400');
    });
    
    // Animate the replay button
    timeline.add({
        targets: '#replay-btn',
        opacity: [0, 1],
        translateY: [10, 0],
        scale: [0.9, 1]
    }, '-=400');
}

// Reset and replay the animation
function replayAnimation() {
    // Reset all elements
    anime.set('.hero-title, .hero-subtitle, .feature-card, #replay-btn', {
        opacity: 0,
        translateY: 20
    });
    
    // Restart the animation
    startAnimation();
}