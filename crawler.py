from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
import time
from src import config as cfg
import pandas as pd
import os


def handle_cookie_consent(driver: webdriver.Firefox) -> None:
    """
    Handle the cookie consent dialog that appears on the website and blocks some of the buttons.
    """
    try:
        # Wait for the cookie consent dialog to appear
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "div#CybotCookiebotDialog"))
        )

        # Wait for the specific button for necessary cookies only to be clickable
        necessary_cookies_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@id='CybotCookiebotDialogBodyButtonDecline']"))
        )

        # Scroll to the button and click it
        driver.execute_script("arguments[0].scrollIntoView(true);", necessary_cookies_button)
        time.sleep(1)
        necessary_cookies_button.click()
        print("Necessary cookies consent given.")
    except TimeoutException:
        print("Cookie consent dialog was not found or was not clickable.")
    except ElementNotInteractableException:
        print("The cookie consent button is not interactable.")


def extract_urls(driver: webdriver.Firefox) -> list[str]:
    """
    Main job of the crawler: Extract all URLs from the current page for webscraping.
    """
    # Find all <a> tags within <div class='col'> that have the class "media-block__link"
    columns = driver.find_elements(By.CSS_SELECTOR, "div.col")
    all_links = []
    for column in columns:
        links = column.find_elements(By.CSS_SELECTOR, 'a.media-block__link')
        for link in links:
            href = link.get_attribute('href')
            if href:  # Ensure href is not None
                all_links.append(href)
    return all_links


def get_all_urls(driver: webdriver.Firefox, max_clicks = 1000) -> list[str]:
    """
    Function workd by clicking the load more button on the website to load more content and extract the URLs.
    The function will click the button until the max_clicks limit is reached or the button is not found.
    After all the clicks, the function will go through each column item and extract the URLs.
    """

    total_urls = []

    click_cnt = 1
    while click_cnt < max_clicks:
        print(f"Clicks: {click_cnt}")

        # Find and click the button to load more content
        try:
            load_more_button = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR,
                                                  'a.btn.btn--lg.section__button[rel="nofollow"][href="javascript:void(0)"]'))
            )
            load_more_button.click()

            # Wait for the page to load new content
            WebDriverWait(driver, 20).until(
                lambda driver: driver.find_element(By.CSS_SELECTOR, "div.col")
            )
        except NoSuchElementException:
            print("Element not found. Check the selector.")
        except TimeoutException:
            break
        except Exception as e:
            print(f"Unhandled exception: {e}")

        click_cnt += 1

    new_urls = extract_urls(driver)
    WebDriverWait(driver, 20).until(
        lambda driver: driver.find_element(By.CSS_SELECTOR, "div.col")
    )

    total_urls.extend(new_urls)
    print(f'Final url count: {len(total_urls)}')

    return total_urls


if __name__ == '__main__':

    options = Options()
    options.binary_location = 'C:\\Program Files\\Mozilla Firefox\\firefox.exe'
    gecko_driver_path = 'geckodriver.exe'
    service = Service(gecko_driver_path)
    driver = webdriver.Firefox(service=service, options=options)

    all_urls = []
    tags = []

    for base_url in cfg.MAIN_URL:
        print(base_url)
        tag = base_url.split('/')[-1]
        driver.get(base_url)

        handle_cookie_consent(driver)

        found_urls = get_all_urls(driver)
        all_urls.extend(found_urls)
        tags.extend([tag for _ in range(len(found_urls))])

    print("Total URLs found:", len(all_urls))
    print("Total Tags found:", len(tags))

    df = pd.DataFrame({
        'Tags': tags,
        'URLs': all_urls
    })

    df.to_csv(os.path.join(cfg.DATA_DIR, 'urls.csv'), index=False)
