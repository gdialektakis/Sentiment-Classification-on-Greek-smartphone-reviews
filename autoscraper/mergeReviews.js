var glob = require("glob");
const fs = require('fs');

var promises = [];
function readJSON(file) {
    return new Promise((resolve, reject) => {
        fs.readFile(file, 'utf8', function (err, data) {
            data = JSON.parse(data);
            resolve(data)
        });
    });
}

glob("reviews/*.json", function(err, files) {
    if(err) {
      console.log("cannot read the folder, something goes wrong with glob", err);
    }
    console.log('\n');
    console.log("Reading reviews");
    console.log("==========================");
    files.forEach(function(file) {
        promises.push(readJSON(file));
    });
    Promise.all(promises).then((reviews) => {
        var all_reviews = [];
        var total = 0;
        var total_with_text = 0;
        for(let i = 0; i < reviews.length; i++) {
            phone_reviews = reviews[i];
            total_with_text += phone_reviews.filter(r => r.review.length > 0).length;
            all_reviews.push(...phone_reviews);
            total += phone_reviews.length;
        }
        console.log("Creating merged file");
        console.log("==========================");
        let jsonString = JSON.stringify(all_reviews, null, 2);
        fs.writeFileSync('./all_reviews.json', jsonString);

        console.log('Total reviews: ' + total);
        console.log('Total reviews with text: ' + total_with_text);
    });
});