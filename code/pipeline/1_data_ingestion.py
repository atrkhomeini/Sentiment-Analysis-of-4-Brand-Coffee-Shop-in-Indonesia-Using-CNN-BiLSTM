from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import re

#---------------------------------------------------------------------------------------------------
# Set up the web driver and firefox
#---------------------------------------------------------------------------------------------------
# Path ke GeckoDriver, sesuaikan dengan path di komputermu
gecko_path = "D:/geckodriver.exe"  # Sesuaikan path ke GeckoDriver
service = Service(executable_path=gecko_path)

# Path ke Firefox executable
firefox_path = "C:/Program Files/Mozilla Firefox/firefox.exe"  # Lokasi Firefox.exe

# Konfigurasi Firefox
options = Options()
options.binary_location = firefox_path  # Menetapkan lokasi binary Firefox
options.headless = False  # Set ke True jika Anda tidak ingin membuka jendela browser

def create_driver():
    return webdriver.Firefox(service=service, options=options)

#---------------------------------------------------------------------------------------------------
# Main function
#---------------------------------------------------------------------------------------------------

def get_tweet(element):
    try:
        user = element.find_element(By.XPATH, ".//*[contains(text(), '@')]").text
        text = element.find_element(By.XPATH, ".//div[@lang]").text
        date = element.find_element(By.XPATH, ".//time").get_attribute("datetime")

        try:
            reply_to_user = element.find_element(By.XPATH, ".//div[@dir='ltr']//a[contains(@href, '/')]").text
        except:
            reply_to_user = None

        try:
            tweet_link = element.find_element(By.XPATH, ".//a[@href and contains(@href, '/status/')]").get_attribute("href")
        except:
            tweet_link = None

        try:
            media_links = [media.get_attribute("src") for media in element.find_elements(By.XPATH, ".//img[contains(@src, 'media')]")]
        except:
            media_links = []

        tweet_data = [user, text, date, tweet_link, media_links, reply_to_user]
        return tweet_data
    except Exception as e:
        print(f"Error: {e}")
        return None

def click_retry_button(driver):
    try:
        retry_button = driver.find_element(By.XPATH, "//div[text()='Retry']/parent::div")
        retry_button.click()
        print("Clicked the retry button")
        WebDriverWait(driver, 15).until(EC.invisibility_of_element_located((By.XPATH, "//div[text()='Retry']")))
    except:
        pass

def scrape_tweets(driver):
    user_data = []
    text_data = []
    date_data = []
    tweet_link_data = []
    media_link_data = []
    reply_to_data = []
    tweet_ids = set()

    scrolling = True
    while scrolling and len(user_data) < 10000:
        try:
            print("Searching for tweets...")
            tweets = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.XPATH, "//article")))
            print(f"Number of tweets found: {len(tweets)}")
            for tweet in tweets:
                tweet_list = get_tweet(tweet)
                if tweet_list:
                    tweet_id = ''.join(tweet_list[:2])
                    if tweet_id not in tweet_ids:
                        tweet_ids.add(tweet_id)
                        user_data.append(tweet_list[0])
                        text_data.append(" ".join(tweet_list[1].split()))
                        date_data.append(tweet_list[2])
                        tweet_link_data.append(tweet_list[3])
                        media_link_data.append(tweet_list[4])
                        reply_to_data.append(tweet_list[5])

            last_height = driver.execute_script("return document.body.scrollHeight")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            WebDriverWait(driver, 30).until(lambda driver: driver.execute_script("return document.body.scrollHeight") > last_height)
            new_height = driver.execute_script("return document.body.scrollHeight")

            click_retry_button(driver)

            if new_height == last_height:
                scrolling = False
            else:
                last_height = new_height
                time.sleep(5)  # Tambahkan penundaan yang lebih lama

        except Exception as e:
            print(f"Error during scrolling or tweet retrieval: {e}")
            break

    return user_data, text_data, date_data, tweet_link_data, media_link_data, reply_to_data

driver = create_driver()
web = 'https://x.com/search?q=foodfess2%20kopken%202024%20lang%3Aid%20until%3A2024-07-01%20since%3A2023-07-01&src=typed_query&f=live' # Masukan link yang akan di cari
driver.get(web)
time.sleep(15)
driver.save_screenshot('screenshot_before_search.png')

user_data, text_data, date_data, tweet_link_data, media_link_data, reply_to_data = scrape_tweets(driver)

driver.save_screenshot('screenshot_after_search.png')
driver.quit()

df = pd.DataFrame({
    'User': user_data,
    'Text': text_data,
    'Date': date_data,
    'Tweet Link': tweet_link_data,
    'Media Links': media_link_data,
    'Reply To': reply_to_data
})

print(df)


#---------------------------------------------------------------------------------------------------
# Extract mentions
#---------------------------------------------------------------------------------------------------
def extract_mentions(text):
    mentions = re.findall(r'@(\w+)', text)
    return mentions

df['Target'] = df['User']
df['Mentions'] = df['Text'].apply(extract_mentions)

#--------------------------------------------------------------------------------
# Save the data
#--------------------------------------------------------------------------------
df.to_csv('hasil-crawling-kopken.csv', index=False, sep=";")