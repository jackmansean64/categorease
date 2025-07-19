/**
 * Client-side handler for batch categorization of transactions
 */

// Initialize socket connection
const socket = io();

// Keep track of how many batches have been processed
let batchCount = 0;
let totalTransactionsProcessed = 0;
let isProcessing = false;

// Connect to the socket server
socket.on('connect', () => {
    console.log('Connected to socket server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from socket server');
});

// Listen for progress bar updates
socket.on('initializeProgressBar', (data) => {
    console.log('Initialize progress bar', data);
    // You can add code here to show a progress bar in the UI
});

socket.on('updateProgressBar', () => {
    console.log('Update progress bar');
    // You can add code here to update the progress bar
});

socket.on('clearProgressBar', () => {
    console.log('Clear progress bar');
    // You can add code here to hide the progress bar
});

socket.on('error', (data) => {
    console.error('Error:', data.error);
    alert('Error: ' + data.error);
});

/**
 * Starts the batch categorization process.
 * This is called from the Excel alert dialog.
 */
function startBatchCategorization(result) {
    if (result === 'cancel') {
        return;
    }

    if (isProcessing) {
        console.log('Already processing a batch. Please wait.');
        return;
    }

    console.log('Starting batch categorization');
    isProcessing = true;
    batchCount = 0;
    totalTransactionsProcessed = 0;

    // Start the first batch
    processNextBatch(false);
}

/**
 * Process the next batch of transactions
 * @param {boolean} fromState - Whether to use the transaction state (true after the first batch)
 */
async function processNextBatch(fromState) {
    try {
        console.log(`Processing batch ${batchCount + 1} (from state: ${fromState})`);

        // Get the current Excel workbook data
        const currentWorkbook = xlwings.getCurrentWorkbookJson();

        // Call the server to process the batch
        const response = await fetch(`/categorize-transaction-batch?from_state=${fromState}&batch_size=10`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentWorkbook),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update the workbook with the processed data
        xlwings.setWorkbookJson(data.workbook);

        // Update counters
        batchCount++;
        totalTransactionsProcessed += data.batch_count;

        // Show progress to the user
        xlwings.showStatus(`Processed ${totalTransactionsProcessed} transactions so far...`);

        // If there are more transactions to process, continue with the next batch
        if (data.has_more) {
            // Short timeout to allow UI updates
            setTimeout(() => {
                processNextBatch(true);
            }, 500);
        } else {
            // All done!
            isProcessing = false;
            xlwings.clearStatus();
            xlwings.alert(`Categorization complete! Processed ${totalTransactionsProcessed} transactions in ${batchCount} batches.`);
        }
    } catch (error) {
        console.error('Error processing batch:', error);
        isProcessing = false;
        xlwings.clearStatus();
        xlwings.alert(`Error processing transactions: ${error.message}`);
    }
} 