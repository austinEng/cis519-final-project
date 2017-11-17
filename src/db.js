
export function initializeDB() {
  console.log('Initializing database');

  return new Promise((resolve, reject) => {
    const createObjectStores = [];

    const openDB = indexedDB.open('HistoryDB', 2);
    openDB.onerror = e => reject(e.target.error);
    openDB.onupgradeneeded = function(e) {
      const db = e.target.result;
      const historyStore = db.createObjectStore('historyStore', { autoIncrement: true });
      const navigationStore = db.createObjectStore('navigationStore', { autoIncrement: true });
      function nothing() { }
      createObjectStores.push(new Promise(historyStore.transaction.oncomplete || nothing, historyStore.transaction.onerror || nothing));
      createObjectStores.push(new Promise(navigationStore.transaction.oncomplete || nothing, navigationStore.transaction.onerror || nothing));
    };
    openDB.onsuccess = e => Promise.all(createObjectStores).then(() => resolve(e.target.result));
  });
}
