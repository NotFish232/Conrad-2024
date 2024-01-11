Module Module1

    Sub Main()

        Dim portList As String = Nothing
        Dim port As String = Nothing
        Dim batteryVoltage As String = Nothing

        portList = GetSerialPortNames()

        Console.WriteLine(portList)
        Console.WriteLine("Enter Serial Port xArm is connected to.")
        port = Console.ReadLine().ToUpperInvariant()
        Console.WriteLine("You entered """ + port + """." + vbCrLf)

        batteryVoltage = GetBatteryVoltage(port)
        Console.WriteLine("Battery voltage: " + batteryVoltage + " Volts." + vbCrLf)

        Console.WriteLine("Homeing xArm servos. Press Enter Key to Begin.")
        Console.ReadLine()
        HomeServos(port)
        Console.WriteLine("Servo homing done." + vbCrLf)

        Console.WriteLine("Press Enter Key to Exit.")
        Console.ReadLine()

    End Sub

    Function GetSerialPortNames() As String

        ' Show all available COM ports.
        Dim portList As String = Nothing

        For Each sp As String In My.Computer.Ports.SerialPortNames
            portList = portList + sp + vbCrLf
        Next

        Return portList

    End Function

    Sub SendSerialData(bytes() As Byte, ByRef com As IO.Ports.SerialPort)

        ' Send byte arrray to a serial port.
        com.Write(bytes, 0, bytes.Length)

    End Sub

    Function ReceiveSerialData(ByRef com As IO.Ports.SerialPort) As Byte()

        ' Receive strings from a serial port.
        Dim bytes(15) As Byte
        Dim length As Integer

        Threading.Thread.Sleep(1000)
        Try
            com.Read(bytes, 0, 4) ' xArm will return 2-byte SIGNATURE, 1 byte length, 1 byte COMMAND, 0 or more bytes of DATA
            If bytes(0) = &H55 And bytes(1) = &H55 Then
                length = bytes(2) - 2
                com.Read(bytes, 0, length)
            End If
        Catch ex As TimeoutException
            Console.WriteLine("Error: Serial Port read timed out.")
        End Try

        If length > 0 Then
            Dim data(length - 1) As Byte
            Array.Copy(bytes, 0, data, 0, length) ' Remove SIGNATURE, LENGTH and COMMAND bytes
            Return data
        End If

        Return Nothing

    End Function

    Function GetBatteryVoltage(port As String) As String

        Dim sendBytes(3) As Byte
        Dim recvBytes(5) As Byte
        Dim millivolts As Integer

        sendBytes(0) = &H55 ' SIGNATURE
        sendBytes(1) = &H55 ' SIGNATURE
        sendBytes(2) = 2    ' Message length (from this byte to the end)
        sendBytes(3) = &HF  ' CMD_GET_BATTERY_VOLTAGE 

        Using com As IO.Ports.SerialPort = My.Computer.Ports.OpenSerialPort(port, 9600)

            com.ReadTimeout = 10000

            SendSerialData(sendBytes, com)
            recvBytes = ReceiveSerialData(com)

        End Using

        millivolts = recvBytes(1) * 256 + recvBytes(0)

        Return (millivolts / 1000).ToString()

    End Function

    Sub HomeServos(port As String)

        Dim sendBytes(24) As Byte

        sendBytes(0) = &H55  ' SIGNATURE
        sendBytes(1) = &H55  ' SIGNATURE
        sendBytes(2) = 23    ' Message length (from this byte to the end)
        sendBytes(3) = &H3   ' CMD_SERVO_MOVE 

        sendBytes(4) = 6     ' Number of motors in list

        sendBytes(5) = &HE8  ' duration low byte (1000 milliseconds, 0x03E8)
        sendBytes(6) = &H3   ' duration high byte

        sendBytes(7) = &H1   ' servo ID 
        sendBytes(8) = &HF4  ' position low byte 
        sendBytes(9) = 1     ' position high byte

        sendBytes(10) = &H2  ' servo ID 
        sendBytes(11) = &HF4 ' position low byte 
        sendBytes(12) = &H1  ' position high byte

        sendBytes(13) = &H3  ' servo ID 
        sendBytes(14) = &HF4 ' position low byte 
        sendBytes(15) = &H1  ' position high byte

        sendBytes(16) = &H4  ' servo ID 
        sendBytes(17) = &HF4 ' position low byte 
        sendBytes(18) = &H1  ' position high byte

        sendBytes(19) = &H5  ' servo ID 
        sendBytes(20) = &HF4 ' position low byte 
        sendBytes(21) = &H1  ' position high byte

        sendBytes(22) = &H6  ' servo ID 
        sendBytes(23) = &HF4 ' position low byte 
        sendBytes(24) = &H1  ' position high byte

        Using com As IO.Ports.SerialPort = My.Computer.Ports.OpenSerialPort(port, 9600)

            com.ReadTimeout = 10000

            SendSerialData(sendBytes, com)

            Threading.Thread.Sleep(2000) ' Allow command t be sent and motion to complete before closing com port

        End Using

    End Sub

End Module
