var glob = require("glob");
const fs = require('fs');
var natural = require('natural');
var TfIdf = natural.TfIdf;

var tfidf = new TfIdf();
const all_tokens = [];
fs.readFile("all_reviews.json", 'utf8', function (err, data) {
    data = JSON.parse(data);
    const reviews_per_rating = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    const tf_per_rating = {"1": [], "2": [], "3": [], "4": [], "5": []}
    // let removeThese = ['\n',',', ',', '.', '!', 'το', 'και', 'τα', 'τη', 'ο', 'η', 'την']
    for(let i = 0; i < data.length; i++) {
        phone_review = data[i];
        reviews_per_rating[phone_review.rating] += 1;
        let current = phone_review.review;
        // removeThese.forEach(r => {
        //     current = current.split(r).join('');
        // });
        let cleaned = current.toLocaleLowerCase();
        tf_per_rating[phone_review.rating].push(...cleaned.split(' '));
        all_tokens.push(...cleaned.split(' '));
    }
    tfidf.addDocument(all_tokens);

    console.log("Stats on reviews");
    console.log("==========================");
    console.log('Total reviews: ' + data.length);
    console.log("==========================");
    console.log("Total Reviews per Rating:");
    console.log(reviews_per_rating);
    console.log("==========================");
    const stats = {
        "total": data.length,
        "per_rating": reviews_per_rating
    }

    let jsonString = JSON.stringify(stats, null, 2);
    fs.writeFileSync('./visualization/stats.json', jsonString);

    // console.log(tfidf)
    for (const [key, value] of Object.entries(tf_per_rating)) {
        let tmpTfIdf = new TfIdf();
        tmpTfIdf.addDocument(value);
        let freq = tmpTfIdf.listTerms(0, true);
        let top = [];
        for(let i = 0; i < 10; i++) {
            let current = freq[i];
            top.push({
                term: current.term,
                occurences: current.tf
            })
        }

        console.log(top);
    }
});