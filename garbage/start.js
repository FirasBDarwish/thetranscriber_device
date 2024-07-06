const { Meet } = require('@shaunbharat/google-meet-api');
const client = new Meet();

const config = {
    meetingLink: 'https://meet.google.com/srq-xbjw-bzd',
    email: 'merch.fye@gmail.com',
    pw: 'Qt/!p"a?vv"[8Vy-'
  };

//   async function sendMessageLoop() {
//     await client.login(config);
  
//     // Send a message every 5 seconds (adjust as needed)
//     setInterval(async () => {
//       try {
//         await client.sendMessage("Hello, World!");
//         console.log('Message sent successfully');
//       } catch (error) {
//         console.error('Error sending message:', error);
//       }
//     }, 5000); // 5000 milliseconds = 5 seconds
//   }
  
//   sendMessageLoop().catch(err => console.error('Error in main loop:', err));

async function command(client, message) {
    if (message.content.startsWith("!quote")) {
        await client.sendMessage(`${message.author} said, "${message.content.replace("!quote ", "")}" at ${message.time}`);
    }

}

(async () => {

    await client.once('ready', async () => {
        console.log('ready');
    })

    await client.login(config);

    await client.on('message', async (message) => {
        command(client, message);
    })

    await client.on('memberJoin', async (member) => {
        await client.sendMessage(`Welcome, ${member.name}!`);
    })

    await client.on('memberLeave', async (member) => {
        await client.sendMessage(`Goodbye, ${member.name}!`);
    })

})()