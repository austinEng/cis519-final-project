import { initializeDB } from './db';
import { initializeLogging } from './logging';

initializeDB().then(db => {
  initializeLogging(db, {
    onInteraction(item) {
      console.log(item);
    },
    onNavigation(item) {
      console.log('Navigate', item);
    },
  });

  window.exportData = function() {
    const historyItems = [];
    const navigationItems = [];

    const transaction = db.transaction(['historyStore', 'navigationStore'], 'readonly');
    const historyStore = transaction.objectStore('historyStore');
    const navigationStore = transaction.objectStore('navigationStore');

    const { processHistoryItem, processNavigationItem } = (() => {
      var currentHistoryCursor = undefined;
      var currentNavigationCursor = undefined;

      function handleNavigation() {
        const item = currentNavigationCursor.value;
        // console.log('Navigation:', item.time, item);
        navigationItems.push(item);
        const temp = currentNavigationCursor;
        currentNavigationCursor = undefined;
        temp.continue();
      }

      function handleHistory() {
        const item = currentHistoryCursor.value;
        // console.log('Interaction:', item.time, item);
        historyItems.push(item);
        const temp = currentHistoryCursor;
        currentHistoryCursor = undefined;
        temp.continue();
      }

      function processInOrder() {
        if (currentHistoryCursor && currentNavigationCursor) {
          if (currentNavigationCursor.value.time <= currentHistoryCursor.value.time) {
            handleNavigation();
          } else {
            handleHistory();
          }
        }

        if (currentHistoryCursor === null && currentNavigationCursor) {
          handleNavigation();
        }

        if (currentNavigationCursor === null && currentHistoryCursor) {
          handleHistory();
        }
      }

      return {
        processHistoryItem(cursor) {
          currentHistoryCursor = cursor;
          processInOrder();
        },
        processNavigationItem(cursor) {
          currentNavigationCursor = cursor;
          processInOrder();
        },
      }
    })();

    const parseHistoryPromise = new Promise((resolve, reject) => {
      historyStore.openCursor().onsuccess = function(e) {
        const cursor = e.target.result;
        if (cursor) {
          processHistoryItem(cursor);
        } else {
          processHistoryItem(null);
          console.log('Finished parsing history items');
          resolve();
        }
      };
    });

    const parseNavigationPromise = new Promise((resolve, reject) => {
      navigationStore.openCursor().onsuccess = function(e) {
        const cursor = e.target.result;
        if (cursor) {
          processNavigationItem(cursor);
        } else {
          processNavigationItem(null);
          console.log('Finished parsing navigation items');
          resolve();
        }
      }
    });

    Promise.all([parseHistoryPromise, parseNavigationPromise]).then(() => {
      const data = {
        navigationItems,
        historyItems,
      };
      const a = document.createElement('a');
      const file = new Blob([JSON.stringify(data)], {type: 'text/plain'});
      a.href = URL.createObjectURL(file);
      a.download = 'browsingData.json';
      a.click();
    });
  }
});
