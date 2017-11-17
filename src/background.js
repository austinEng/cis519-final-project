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

  const transaction = db.transaction(['historyStore', 'navigationStore'], 'readonly');
  const historyStore = transaction.objectStore('historyStore');
  const navigationStore = transaction.objectStore('navigationStore');

  const { processHistoryItem, processNavigationItem } = (() => {
    var currentHistoryCursor = undefined;
    var currentNavigationCursor = undefined;

    function handleNavigation() {
      const item = currentNavigationCursor.value;
      console.log('Navigation:', item.time, item);
      const temp = currentNavigationCursor;
      currentNavigationCursor = undefined;
      temp.continue();
    }

    function handleHistory() {
      const item = currentHistoryCursor.value;
      console.log('Interaction:', item.time, item);
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

  historyStore.openCursor().onsuccess = function(e) {
    const cursor = e.target.result;
    if (cursor) {
      processHistoryItem(cursor);
    } else {
      processHistoryItem(null);
      console.log('Finished parsing history items');
    }
  };

  navigationStore.openCursor().onsuccess = function(e) {
    const cursor = e.target.result;
    if (cursor) {
      processNavigationItem(cursor);
    } else {
      processNavigationItem(null);
      console.log('Finished parsing navigation items');
    }
  }
});
