---
sidebar_label: Important Questions on Computer Networks
title: Important Questions on Computer Networks
sidebar_position: 12
---

:::info Source
[TakeUforward](https://takeuforward.org/interviews/must-do-questions-for-dbms-cn-os-interviews-sde-core-sheet/)
:::

1. What is a network?
   
   A network is a set of devices that are connected with a physical media link. 
   
   In a network, two or more nodes are connected by a physical link or two or more networks are connected by one or more nodes. 
   
   A network is a collection of devices connected to each other to allow the sharing of data.

2. What is network topology?
   
   Network topology is the arrangement of nodes and links of a network.

   Topologies are either physical or logical network topology.

   Network topology can be categorized into – 
   - Bus Topology, 
   - Ring Topology, 
   - Star Topology, 
   - Mesh Topology, 
   - Tree Topology.

   [More](https://takeuforward.org/computer-network/what-is-network-and-network-topology/) on network and topology

3. What is bandwidth, node and link?
   
   Bandwidth is the data transfer capacity of a computer network in bits per second (Bps).

   A **network** is a connection setup of two or more computers directly connected by some physical mediums like optical fibre or coaxial cable. This physical medium of connection is known as a **link**, and the computers that it is connected to are known as **nodes**

4. What is OSI model?
   
   The Open System Interconnection (OSI) model is a conceptual model developed by the International Standards Organization (ISO) in 1984. The OSI model provides a standard for communication between different/diverse computer systems.

   The OSI model has seven layers in which each layer has a specific set of functions and communicates with the layer above and below itself.

5. Explaination on TCP model.
   
   It is a compressed version of the OSI model with only 4 layers. 
   
   It was developed by the US Department of Defence (DoD) in the 1860s. 
   
   The name of this model is based on 2 standard protocols used i.e. **TCP (Transmission Control Protocol)** and **IP (Internet Protocol)**.

   **Network Access/Link layer** : Decides which links such as serial lines or classic Ethernet must be used to meet the needs of the connectionless internet layer. Ex – Sonet, Ethernet

   **Internet** : The internet layer is the most important layer which holds the whole architecture together. It delivers the IP packets where they are supposed to be delivered. Ex – IP, ICMP.

   **Transport** : Its functionality is almost the same as the OSI transport layer. It enables peer entities on the network to carry on a conversation. Ex – TCP, UDP (User Datagram Protocol)

   **Application** : It contains all the higher-level protocols. Ex – HTTP, SMTP, RTP, DNS

6. What are the layers of OSI model?
   
   **Physical layer:**

   - The lowest layer of OSI model.
   - Responsible for transmitting message bits over a medium and takes care of mechanical, electrical, procedural and functional specifications for communication.

   Functions:
   - Transmission mode: It defines a transmission mode from Simplex, half-duplex, and full-duplex.
   - Network Topology: It specifies the arrangement of devices in a network.
   - Physical characteristics of  the transmission medium
   - Line Configuration: It selects from either point-to-point or multipoint line configuration.
   - Data Rate: The physical layer defines the number of bits transmitted per unit of time.

   **Data Link layer**:

   - This layer data into packets received from network layer into smaller pieces called frames.

   The DLL is divided into two sublayers:
   - LLC(Logical Link Control): It deals with functions like flow control and error control.
   - MAC(Media Access Control): It controls the physical addressing and framing functions of the data link layer.

   Functions:

   - Flow Control: It makes sure that the transmitting speed and the amount of data sent match with the capacity and speed of the receiver so that no data gets corrupted.
   - Framing: DLL adds certain bits at the beginning(called header which contains the source and destination addresses) and at the end(called trailer which contains error correction and detection bits) to the message frame. 
   - Error Control: DLL uses CRC(cyclic redundancy check) to check if any error occurred during transmission.
   - Physical Addressing: DLL adds physical address(MAC address) of destination and source in the header of each frame.
   - Access Control: Determines which device has control over the link if the same communication channel is shared by multiple devices.

7. What is the significance of Data Link layer?

   - It is used for transferring the data from one node to another node.
   - It receives the data from the network layer and converts the data into data frames and then attaches the physical address to these frames which are sent to the physical layer.
   - It enables the error-free transfer of data from one node to another node.

8. What is gateway?

   A node that is connected to two or more networks is commonly known as a gateway.

   It is also known as a router. It is used to forward messages from one network to another.

9. Difference between gateway and router.
    
   A router sends the data between two similar networks while gateway sends the data between two dissimilar networks.

10. What is DNS?

      **DNS** is an acronym that stands for Domain Name System.DNS was introduced by Paul Mockapetris and Jon Postel in 1983.

      It is a naming system for all the resources over the internet which includes physical nodes and applications. It is used to locate resources easily over a network.

      DNS is an internet which maps the domain names to their associated IP addresses. Without DNS, users must know the IP address of the web page that you wanted to access.

11. What is DNS forwarder?

      A forwarder is used with a DNS server when it receives DNS queries that cannot be resolved quickly. So it forwards those requests to external DNS servers for resolution. A DNS server which is configured as a forwarder will behave differently than the DNS server which is not configured as a forwarder.

12. What is NIC?

      NIC stands for Network Interface Card. It is a peripheral card attached to the PC to connect to a network.

      Every NIC has its own MAC address that identifies the PC on the network. It provides a wireless connection to a local area network. 
   
      NICs were mainly used in desktop computers.

13. What is MAC address?

   
      A media access control address (MAC address) is a unique identifier assigned to a network interface controller (NIC) for use as a network address in communications within a network segment.

14. What is IP address, private IP, public IP and APIPA?

      An IP address is a unique address that identifies a device on the internet or a local network.

      IP stands for “Internet Protocol,” which is the set of rules governing the format of data sent via the internet or local network.

      **Private IP Address** – There are three ranges of IP addresses that have been reserved
for IP addresses. They are not valid for use on the internet. If you want to access the
internet on these private IPs, you must use a proxy server or NAT server.

      **Public IP Address** – A public IP address is an address taken by the Internet Service
Provider which facilitates communication on the internet.

      **APIPA** stands for Automatic Private IP Addressing (APIPA). It is a feature or characteristic in operating systems (eg. Windows) which enables computers to self-configure an IP address and subnet mask automatically when their DHCP server isn’t reachable.

15. What is DHCP?
    
      A DHCP or Dynamic Host Configuration Protocol Server, is a network server that automatically provides and assigns IP addresses, default gateways and other network parameters to client devices. 
      
      It relies on the standard protocol known as Dynamic Host Configuration Protocol

16. What is IPv4 and IPv6
17. What is a subnet?

      A subnet is a network inside a network achieved by the process called subnetting which helps divide a network into subnets. 
      
      It is used for getting a higher routing efficiency and enhances the security of the network. 
      
      It reduces the time to extract the host address from the routing table.

18. What are firewalls?

      The firewall is a network security system that is used to monitor the incoming and outgoing traffic and blocks the same based on the firewall security policies. 

      It acts as a wall between the internet (public network) and the networking devices (a private network). 
      
      It is either a hardware device, software program, or a combination of both. 
      
      It adds a layer of security to the network.

19. What is 3 way handshaking?

      Three-Way HandShake or a TCP 3-way handshake is a process which is used in a TCP/IP network to make a connection between the server and client. 
      
      It is a three-step process that requires both the client and server to exchange synchronisation and acknowledgment packets before the real data communication process starts.

      Three-way handshake process is designed in such a way that both ends help you to initiate, negotiate, and separate TCP socket connections at the same time. 
      
      It allows you to transfer multiple TCP socket connections in both directions at the same time.

20. What is server-side load balancer?

      All backend server instances are registered with a central load balancer. 
      
      A client requests this load balancer which then routes the request to one of the server instances using various algorithms like round-robin. 
      
      AWS ELB(Elastic Load Balancing) is a prime example of server-side load-balancing that registers multiple EC2 instances launched in its auto-scaling group and then routes the client requests to one of the EC2 instances.

      Advantages of server-side load balancing:

      - Simple client configuration: only need to know the load-balancer address.
      - Clients can be untrusted: all traffic goes through the load-balancer where it can be looked at. Clients are not aware of the backend servers.

21. What is RSA algorithm?

      RSA algorithm is an asymmetric cryptography algorithm. Asymmetric actually means that it works on two different keys i.e. Public Key and Private Key. As the name describes, the Public Key is given to everyone and the Private key is kept private.

22. What is HTTP and HTTPS?

      HTTP is the **HyperText Transfer Protocol** which defines the set of rules and standards on how the information can be transmitted on the World Wide Web (WWW). It helps the web browsers and web servers for communication. It is a ‘stateless protocol’ where each command is independent with respect to the previous command. 
      
      HTTP is an application layer protocol built upon the TCP. It uses port 80 by default. 
      
      HTTPS is the HyperText Transfer Protocol Secure or Secure HTTP. It is an advanced and a secured version of HTTP. 
      
      On top of HTTP, SSL/TLS protocol is used to provide security. It enables secure transactions by encrypting the communication and also helps identify network servers securely. It uses port 443 by default

23. What is SMTP?

      SMTP is the **Simple Mail Transfer Protocol** which sets the rule for communication between servers. This set of rules helps the software to transmit emails over the internet. 
      
      It supports both End-to-End and Store-and-Forward methods. It is in always-listening mode on port 25.

24. What is TCP and UDP?

      TCP is a connection-oriented protocol, whereas UDP is a connectionless protocol. 
      
      A key difference between TCP and UDP is speed, as TCP is comparatively slower than UDP. Overall, UDP is a much faster, simpler, and efficient protocol, however, retransmission of lost data packets is only possible with TCP.

      TCP provides extensive error checking mechanisms. It is because it provides flow control and acknowledgment of data. UDP has only the basic error checking mechanism using checksums.

25. What happends when we enter 'google.com'?

      1. Check the browser cache first if the content is fresh and present in the cache display the same.
      2. If not, the browser checks if the IP of the URL is present in the cache (browser and OS) if not then requests the OS to do a DNS lookup using UDP to get the corresponding IP address of the URL from the DNS server to establish a new TCP connection.
      3. A new TCP connection is set between the browser and the server using three-way handshaking.
      4. An HTTP request is sent to the server using the TCP connection.
      5. The web servers running on the Servers handle the incoming HTTP request and send the HTTP response.
      6. The browser processes the HTTP response sent by the server and may close the TCP connection or reuse the same for future requests.
      7. If the response data is cacheable then browsers cache the same.
      8. Browser decodes the response and renders the content.

