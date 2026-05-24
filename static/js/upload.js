function handleInput(event){
    // const file = event.target.files[0];
    // const reader = new FileReader();

    // reader.onload = function(e) {
    //     const inputImage = document.getElementById('input-image');
    //     inputImage.src = e.target.result;
    // };

    // reader.readAsDataURL(file);
    console.log("hello");
    const uploaded_info = document.getElementsByClassName("uploaded")[0]
    uploaded_info.style.visibility = 'visible';
}


function handleFileUpload(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const inputImage = document.getElementById('result-image');
        inputImage.src = e.target.result;
    };

    reader.readAsDataURL(file);

    // event.preventDefault();
    // console.log("HEllo");
    // const file = event.target.files[0];

    // const formData = new FormData();
    // formData.append("image", file);

    // fetch('/predict', {
    //     method: 'POST',
    //     body: formData
    // })
    // .then(response => response.json())
    // .then(data => {
    //     const processedImageURL = data.processed_image_url;
    //     const imgElement = document.getElementById("result-image");
    //     imgElement.src = `data:image/png;base64,${processedImageURL}`;
    // })
    // .catch(error => console.error('Error:', error));
}


function handleFileUpload_trial()
{
    console.log("try");
}