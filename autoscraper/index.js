const puppeteer = require('puppeteer');
const fs = require('fs');

(async ee => {
    const links = [

    ];
    console.log('Start Scraper');
    const browser = await puppeteer.launch({ headless: true, slowMo: 100, devtools: true });

    const page = await browser.newPage();
    for(let i = 0; i < links.length; i++) {
        const link = links[i];
        await page.goto(link);
        try {
            var min = 2;
            var max = 5;
            for(let i = 0; i < 100000; i++) {
                console.log(link + ' - ' + i)
                var rand = Math.floor(Math.random() * (max - min + 1) + min);
                await page.waitForSelector(".load-more-reviews");
                const btn = await page.$('.load-more-reviews');
                await btn.evaluate(btn => btn.click());
                await page.waitForTimeout(rand * 1000);
            }
        } catch (error) {
            console.log("Start fetching stuff");
            let phone = await page.evaluate(() => document.querySelector('.page-title').innerText);
            let file_name = phone.toLowerCase().split(' ').join('_').split('(').join('').split(')').join('').split('-').join('').split(',').join('').split('.').join('').split('+').join('').split('/').join('');
            const reviews = await page.evaluate(() => 
                Array.from(document.querySelectorAll('.review-item'), item => {
                    phone = document.querySelector('.page-title').innerText;
                    const rating = item.querySelector('.actual-rating span').innerText;
                    const review = item.querySelector('.review-body').innerText;
        
                    const all_pros = Array.from(item.querySelectorAll('.icon.pros li'), li => li.innerText);
                    const all_soso = Array.from(item.querySelectorAll('.icon.so-so li'), li => li.innerText);
                    const all_cons = Array.from(item.querySelectorAll('.icon.cons li'), li => li.innerText);
                    return {
                        phone: phone,
                        rating: rating,
                        review: review,
                        pros: all_pros,
                        neutral: all_soso,
                        cons: all_cons,
                    }
                })
            );
            let jsonString = JSON.stringify(reviews, null, 2);
            fs.writeFileSync('./reviews/' + file_name + '.json', jsonString);
        }
    }
    
    await browser.close();
    console.log('Finished all links :)');
})();