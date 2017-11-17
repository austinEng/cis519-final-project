import LRUCache from 'lru-cache';

export function initializeLogging(db, params) {
  console.log('Initializing navigation logging');
  // chrome.webNavigation.onBeforeNavigate.addListener(e => {
  //     console.log('onBeforeNavigate', e);
  // });

  var transitionType, transitionQualifiers;

  chrome.webNavigation.onCommitted.addListener(e => {
      transitionType = e.transitionType;
      transitionQualifiers = e.transitionQualifiers;
  });

  // chrome.webNavigation.onDOMContentLoaded.addListener(e => {
  //     console.log('onDOMContentLoaded', e);
  // });

  // chrome.webNavigation.onCompleted.addListener(e => {
  //     console.log('onCompleted', e);
  // });

  // chrome.webNavigation.onErrorOccurred.addListener(e => {
  //     console.log('onErrorOccurred', e);
  // });

  // chrome.webNavigation.onCreatedNavigationTarget.addListener(e => {
  //     console.log('onCreatedNavigationTarget', e);
  // });

  // chrome.webNavigation.onReferenceFragmentUpdated.addListener(e => {
  //     console.log('onReferenceFragmentUpdated', e);
  // });

  // chrome.webNavigation.onTabReplaced.addListener(e => {
  //     console.log('onTabReplaced', e);
  // });

  // chrome.webNavigation.onHistoryStateUpdated.addListener(e => {
  //     console.log('onHistoryStateUpdated', e);
  // });

  const visitCache = new LRUCache({
    max: 50,
    stale: true,
  });

  var fromNewTabPage = undefined;

  function logPageInteraction(time, historyItem) {
    const item = Object.assign({}, historyItem, {
      time,
    });

    const historyStore = db.transaction('historyStore', 'readwrite').objectStore('historyStore');
    const tx = historyStore.add(item);
    tx.onerror = e => console.error(e.target.error);
    tx.onsuccess = () => {
      params.onInteraction && params.onInteraction(item);
    };
  }

  function logNavigation(time, historyItem) {
    const item = Object.assign({}, historyItem, {
      time,
    });

    const navigationStore = db.transaction('navigationStore', 'readwrite').objectStore('navigationStore');
    const tx = navigationStore.add(item);
    tx.onerror = e => console.error(e.target.error);
    tx.onsuccess = () => {
      params.onNavigation && params.onNavigation(item);
    };
  }

  chrome.history.onVisited.addListener(historyItem => {
    const time = performance.timing.navigationStart + performance.now();

    chrome.history.getVisits({
      url: historyItem.url,
    }, visitItems => {
      if (fromNewTabPage === historyItem.url) {
        logNavigation(time, historyItem);
      } else if (transitionType === 'generated' && transitionQualifiers.indexOf('from_address_bar') >= 0) {
        logNavigation(time, historyItem);
      }
      fromNewTabPage = undefined;
      logPageInteraction(time, historyItem);
    });
  });

  chrome.tabs.onCreated.addListener(e => {
      console.log('Predicting:', getPredictions());
  });

  chrome.tabs.onActivated.addListener(e => {
    fromNewTabPage = undefined;
    chrome.tabs.get(e.tabId, tab => {
      const url = new URL(tab.url);

      if (url.protocol == 'chrome:') {
        return;
      }

      const time = performance.timing.navigationStart + performance.now();

      chrome.history.getVisits({
        url: tab.url
      }, visitItems => {
        const id = visitItems[0].id;
        if (visitCache.has(id)) {
          logPageInteraction(time, visitCache.get(id));
        } else {
          chrome.history.search({
            text: tab.url,
            maxResults: 1
          }, historyItems => {
            logPageInteraction(time, historyItems[0]);
          });
        }
      });
    });
  });

  function getPredictions() {
    console.log('Current time:', performance.timing.navigationStart + performance.now());
    console.log('Recent items', visitCache.values().slice(0, 5));
    return [
      {id: 0, url: 'http://www.facebook.com?thing=2&a=4'},
      {id: 1, url: 'https://inbox.google.com/u/0/'},
      {id: 2, url: 'http://localhost:3000'}
    ];
  }

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch(request.type) {
      case 'NAVIGATION_COMMIT':
      fromNewTabPage = request.url;
      sendResponse({
          data: {}
        });
        break;

      case 'INITIAL_SUGGESTIONS':
        sendResponse({
          data: getPredictions(),
        });
        break;

      case 'QUERY_SUGGESTIONS':
        sendResponse({
          data: [
            {id: 0, url: 'http://www.facebook.com?thing=2&a=4'},
            {id: 1, url: 'https://inbox.google.com/u/0/'},
            {id: 2, url: 'http://localhost:3000'}
          ]
        });
        break;

      default:
        sendResponse({
          err: 'Invalid command',
        });
        break;
    }
  });
}
