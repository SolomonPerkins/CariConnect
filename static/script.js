// Handle button click
document.getElementById('submitButton').addEventListener('click', async () => {

const title = document.getElementById('title').value;
const subjects = document.getElementById('subjects').value;
const synopsis = document.getElementById('synopsis').value;


if(!title || !subjects || !synopsis){
    alert('Please provide valid input.')
    return;
}
try {
const response = await fetch('http://127.0.0.1:8000/recommend', {
    method: 'POST',
    headers: {
    'Content-Type': 'application/json'
    },
    body: JSON.stringify({ title, subjects, synopsis })
});

const data = await response.json(); 
// console.log("publishers", JSON.stringify(data, null, 2))

//data (dict obj) needed is here...sample
//{
//   "recommendations": [
//     {
//       "index": 5999,
//       "publisher": "TokyoPop",
//       "similarity_score": 0,
//       "subjects": "Children's Books\nLiterature & Fiction\nComics & Graphic Novels\nManga",
//       "title": "Pirates of the Caribbean: Dead Man's Chest Dead Man's Chest: Island of the Peleg"
//     },
//     {
//       "index": 2002,
//       "publisher": "University of Virginia Press",
//       "similarity_score": 0,
//       "subjects": "Literature & Fiction\nHistory & Criticism\nRegional & Cultural",
//       "title": "The Purloined Islands: Caribbean-U.S. Crosscurrents in Literature and Culture, 1880?1959 (New World Studies)"
//     },
//}

const publishers = data.recommendations.map(item => item.publisher)
// console.log("publishers", publishers)

if (response.ok) 
{
    // Display results
    const resultsDiv = document.getElementById('results');

    //clear previous results
    // resultsDiv.innerHTML = ''
    

    console.log("RESULTS", JSON.stringify(data, null, 2))
    
    // console.log(document.getElementById('results'));

    if (data.recommendations.length === 0) {
        resultsDiv.innerHTML = '<p>No recommendations found.</p>';
        return;
      }
  
      // Dynamically render recommendations
      data.recommendations.forEach(item => {
        const recommendationHTML = `
          <div class="recommendation">
            <p> ${item.publisher}</p>
          </div>
        `;
        resultsDiv.innerHTML += recommendationHTML;
      });
        
        // Show the results container
        resultsDiv.style.display = 'block';
} else {
    console.error(data.error);
    alert('Error: ' + data.error);
}

} catch (error) {
    console.error('Error fetching recommendations:', error);
    alert('Something went wrong!');
}
});


// Reset Button Logic
document.getElementById('resetButton').addEventListener('click', () => {
    // Clear input fields
    document.getElementById('title').value = '';
    document.getElementById('subjects').value = '';
    document.getElementById('synopsis').value = '';
  
    // Clear and hide results
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    resultsDiv.style.display = 'none';
});
